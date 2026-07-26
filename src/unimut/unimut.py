"""unimut: a small, focused mutation-testing CLI.

unimut looks inside ``// unimut on`` / ``// unimut off`` markers in a
source file, generates every mutation of the code it finds there, and for
each mutation runs a user-supplied shell command (typically "build and run
the relevant tests"). Any mutant the command still passes against is a
*survivor*: a change your test suite failed to notice, which usually means
that code path is under-tested. Mutants that make the command fail --
whether because they broke compilation or because a test actually caught
the change -- are killed, and by default aren't shown at all.

Usage:

    unimut --file src/lj_ffrecord.c --run 'make -j$(nproc) && PATH="$PWD/src:$PATH" perl t/unpack.t'

Three ways to decide what gets mutated, which combine into two typical CI
modes:

* Default (marker-based): only ``// unimut on`` / ``// unimut off``
  regions are mutated. Cheap and precise, but requires markers to be
  present.
* ``--diff <ref>`` (a fast PR gate): scans the whole file, but only keeps
  mutants that land on a line that differs from ``<ref>`` according to
  ``git diff <ref>...HEAD``. A change only has to prove the tests around
  the lines it touched are solid, not the whole file.
* ``--whole-file`` (a slow nightly audit): scans the whole file
  exhaustively. Any ``// unimut off`` / ``// unimut on`` pair still
  present is read *inverted* in this mode: it marks a range to exclude
  (e.g. an allocator-failure branch nothing can reliably trigger), rather
  than a range to include.

``--diff`` implies whole-file scanning (it needs to see the whole file to
know which statements land on changed lines), and honors the same
marker-inverted exclusions as ``--whole-file`` if markers are present.

``--keep-call NAME`` (repeatable) tells unimut never to propose removing
a statement that is nothing but a call to ``NAME`` -- e.g.
``--keep-call printf --keep-call assert`` stops it from reporting your
logging and assertion calls as "untested".

unimut never mutates ``--file`` in place. It always copies the whole
repository (as reported by ``git rev-parse --show-toplevel``, or the
current directory if that isn't a git checkout) into an isolated temp
directory, and mutates and tests that copy instead -- so the real
project on disk is never touched.

``--jobs N`` (default 1) runs N mutants at a time, each in its own
worker *process* with its own throwaway repository copy. This has to be
processes rather than threads: applying a mutant reparses the region
with pycparser, which -- like any pure-Python, CPU-bound work -- holds
the GIL, so threads would mostly serialize on that step no matter how
many cores are free, and building/running the actual mutant is a
separate subprocess either way. Worker *processes* sidestep the GIL
entirely and scale with real CPU cores.

Every run also submits one extra unit of work to that same pool: a
baseline check that builds/tests the code completely unmodified. It's
not run serially up front -- it's just another job sharing the worker
pool, so it costs no extra wall time in the common case where it passes.
If it comes back failing, though, every mutant result is meaningless
(a broken build "survives" any mutation trivially), so unimut cancels
whatever mutant work hasn't started yet, skips the usual report, and
exits with the baseline's captured output instead.

Language support is dispatched by file extension (or ``--lang``) to a
separate backend module. Right now that's just ``mutate_c`` for C; see
``_LANGUAGES`` below for how to add more.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import multiprocessing
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from . import mutate_c

# Maps a --lang value to the backend module that implements it. Each
# backend module must expose:
#   EXTENSIONS: set[str]                          -- recognized file extensions
#   generate_mutants(file_path, source, *, whole_file=False,
#                     changed_lines=None, keep_calls=None) -> list[Mutant]
# where a Mutant has .file, .line, .original, .mutated (str | None) and
# an .apply(source) -> str method that must be picklable (--jobs ships
# mutants to worker processes). The whole_file/changed_lines/keep_calls
# keyword arguments are optional -- a backend that only supports
# marker-based regions can omit them, and unimut will refuse
# --diff/--whole-file/--keep-call for that backend with a clear error
# rather than silently ignoring the flag.
_LANGUAGES = {
    "c": mutate_c,
}


def _detect_language(file_path: Path, lang_override: Optional[str]):
    if lang_override is not None:
        try:
            return _LANGUAGES[lang_override]
        except KeyError:
            known = ", ".join(sorted(_LANGUAGES))
            raise SystemExit(
                f"unimut: error: unknown --lang '{lang_override}' (known: {known})"
            )
    ext = file_path.suffix
    for module in _LANGUAGES.values():
        if ext in module.EXTENSIONS:
            return module
    raise SystemExit(
        f"unimut: error: cannot determine language for '{file_path}' "
        f"(unrecognized extension '{ext}'); pass --lang to override"
    )


def _use_color(stream) -> bool:
    return stream.isatty()


def _red(text: str, color: bool) -> str:
    return f"\033[31m{text}\033[0m" if color else text


def _green(text: str, color: bool) -> str:
    return f"\033[32m{text}\033[0m" if color else text


class MutantResult:
    __slots__ = ("mutant", "survived")

    def __init__(self, mutant, survived: bool):
        self.mutant = mutant
        self.survived = survived


def _run_command(run_cmd: str, cwd: Optional[Path] = None) -> bool:
    """Run the user's --run command; True if it exited 0 (mutant survived)."""
    result = subprocess.run(
        run_cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _git_repo_root(near: Path) -> Optional[Path]:
    """Return the git repo root containing ``near``, or None if there isn't one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=near if near.is_dir() else near.parent,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


# Matches a unified-diff hunk header, e.g. "@@ -12,3 +12,4 @@". Only the
# "new file" side (+) is captured -- that is the side whose line numbers
# match the file's current content, which is what unimut mutates.
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _changed_lines(file_path: Path, diff_ref: str) -> Set[int]:
    """Line numbers in ``file_path``'s current content that differ from ``diff_ref``.

    Implements the ``--diff`` PR-gate mode: shells out to
    ``git diff --unified=0 <diff_ref>...HEAD -- <file_path>`` and parses
    the hunk headers to recover exactly which lines were added or
    modified, without pulling in a full diff-parsing library.

    Requires ``file_path`` to live inside a git repository with
    ``diff_ref`` resolvable (a local branch, remote branch, tag, or
    commit). Raises ``SystemExit`` with a clear message otherwise.
    """
    if shutil.which("git") is None:
        raise SystemExit(
            "unimut: error: --diff requires git, which was not found on PATH"
        )
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{diff_ref}...HEAD", "--", str(file_path)],
        cwd=file_path.resolve().parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"unimut: error: 'git diff {diff_ref}...HEAD -- {file_path}' failed: "
            f"{result.stderr.strip()}"
        )
    changed: Set[int] = set()
    for line in result.stdout.splitlines():
        m = _HUNK_HEADER_RE.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = 1 if m.group(2) is None else int(m.group(2))
        if count == 0:
            # A pure deletion on the new-file side adds nothing to mutate.
            continue
        changed.update(range(start, start + count))
    return changed


# Per-worker-process state, set up once by _init_worker and reused across
# every mutant that worker processes -- so the (potentially large) repo
# copy happens once per worker, not once per mutant.
_worker_workdir: Optional[Path] = None
_worker_target: Optional[Path] = None


def _init_worker(repo_root: str, rel_path: str, workdir_registry) -> None:
    """ProcessPoolExecutor initializer: give this worker its own repo copy.

    Runs once per worker process. Copies ``repo_root`` (skipping ``.git``)
    into a fresh temp directory and remembers where the mutated file
    lives inside it, so later calls to :func:`_run_one_mutant` in this
    same process don't have to.

    ``workdir_registry`` is a ``multiprocessing.Manager`` list shared with
    the parent process; the worker's temp directory is never cleaned up
    from inside the worker itself. When a worker process exits -- whether
    normally, killed, or crashed -- there's no reliable hook (``atexit``
    included) that's guaranteed to run there, so the parent process
    cleans up every directory this registry ends up holding once the
    whole pool has shut down (see :func:`_run_mutants`).
    """
    global _worker_workdir, _worker_target
    workdir = Path(tempfile.mkdtemp(prefix="unimut-worker-"))
    shutil.copytree(
        repo_root, workdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git")
    )
    _worker_workdir = workdir
    _worker_target = workdir / rel_path
    workdir_registry.append(str(workdir))


def _run_one_mutant(mutant, original_source: str, run_cmd: str) -> bool:
    """Runs inside a worker process. Returns True if the mutant survived."""
    assert _worker_workdir is not None and _worker_target is not None
    mutated_source = mutant.apply(original_source)
    _worker_target.write_text(mutated_source)
    try:
        return _run_command(run_cmd, cwd=_worker_workdir)
    finally:
        _worker_target.write_text(original_source)


def _run_baseline(run_cmd: str) -> Tuple[bool, str]:
    """Runs inside a worker process: build/test the code *unmodified*.

    A worker's copy already matches the original source when this runs
    (freshly copied, or restored after any earlier mutant this same
    worker processed), so there's nothing to write first. Captures
    combined stdout/stderr, unlike mutant runs, since a baseline failure
    is worth explaining to the person running unimut.
    """
    assert _worker_workdir is not None
    result = subprocess.run(
        run_cmd,
        shell=True,
        cwd=_worker_workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode == 0, result.stdout


def _run_mutants(
    file_path: Path,
    original_source: str,
    mutants: List,
    run_cmd: str,
    jobs: int,
    repo_root: Path,
) -> Tuple[bool, str, List[MutantResult]]:
    """Run a baseline check plus every mutant across ``jobs`` worker processes.

    Returns ``(baseline_survived, baseline_output, results)``. The
    baseline is a single extra unit of work -- "build/test the code with
    no mutation applied" -- submitted to the very same worker pool as the
    mutants, so it runs *alongside* them rather than serially in front of
    them. If it comes back failing, that means ``--run`` doesn't even
    pass against unmodified code, so no mutant result can be trusted;
    unimut cancels whatever mutant work hasn't started yet and returns
    immediately with ``results`` empty, rather than finishing a run whose
    conclusions would be meaningless.

    unimut never mutates ``--file`` (or anything else in the user's real
    working tree) in place. Each worker process gets its own full copy of
    ``repo_root`` in a temp directory (see :func:`_init_worker`), mutates
    *that* copy's version of the file, and runs ``run_cmd`` there -- so
    the original project is left untouched no matter how ``--run``
    behaves, and workers can't race each other writing to a shared file.

    This uses *processes*, not threads: applying a mutant re-parses the
    marked region with pycparser, which is CPU-bound pure Python and
    therefore holds the GIL, so threads would serialize on that step
    however many cores are idle. Separate processes don't share a GIL, so
    they actually scale with ``--jobs``.
    """
    resolved_root = repo_root.resolve()
    rel_path = file_path.resolve().relative_to(resolved_root)

    results: List[Optional[MutantResult]] = [None] * len(mutants)
    # Filled in by _record_baseline as soon as the baseline future
    # completes -- checked from the mutant loop below so a failing
    # baseline can cut the run short without waiting on it directly
    # (which would serialize it in front of the mutants again).
    baseline_holder: List[Optional[Tuple[bool, str]]] = [None]

    def _record_baseline(fut: "concurrent.futures.Future[Tuple[bool, str]]") -> None:
        baseline_holder[0] = fut.result()

    with multiprocessing.Manager() as manager:
        workdir_registry = manager.list()
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_worker,
                initargs=(str(resolved_root), str(rel_path), workdir_registry),
            ) as executor:
                baseline_future = executor.submit(_run_baseline, run_cmd)
                baseline_future.add_done_callback(_record_baseline)

                mutant_futures = {
                    executor.submit(
                        _run_one_mutant, mutant, original_source, run_cmd
                    ): idx
                    for idx, mutant in enumerate(mutants)
                }
                try:
                    for future in concurrent.futures.as_completed(mutant_futures):
                        idx = mutant_futures[future]
                        results[idx] = MutantResult(mutants[idx], future.result())
                        baseline_result = baseline_holder[0]
                        if baseline_result is not None and not baseline_result[0]:
                            # No point finishing (or even starting) the
                            # rest of the mutants: their results would be
                            # meaningless if --run doesn't even pass
                            # unmodified. Best-effort-cancel whatever
                            # hasn't started yet and stop collecting.
                            for f in mutant_futures:
                                f.cancel()
                            break
                    # The mutant loop may finish (or bail) before the
                    # baseline does, or there may be nothing in it at all
                    # to trigger the check above -- either way, block
                    # here until we actually know the baseline's outcome.
                    baseline_survived, baseline_output = baseline_future.result()
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
            # All worker processes have now exited (the executor's
            # __exit__ waits for that), so every temp directory they
            # created is safe to remove.
            for workdir in workdir_registry:
                shutil.rmtree(workdir, ignore_errors=True)
        except BaseException:
            for workdir in workdir_registry:
                shutil.rmtree(workdir, ignore_errors=True)
            raise

    if not baseline_survived:
        return False, baseline_output, []
    return True, "", [r for r in results if r is not None]


def _print_report(
    results: List[MutantResult], include_killed: bool, color: bool
) -> int:
    shown = [r for r in results if r.survived or include_killed]
    for i, r in enumerate(shown):
        if i > 0:
            print()

        m = r.mutant
        print(f"{m.file}:{m.line}")
        print(_red(f"- {m.original}", color))
        if m.mutated is not None:
            print(_green(f"+ {m.mutated}", color))

    survived_count = sum(1 for r in results if r.survived)

    if shown:
        print()

    print(f"Survived: {survived_count}/{len(results)}")
    return 1 if survived_count > 0 else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unimut",
        description=(
            "Mutation-test the code between '// unimut on' and "
            "'// unimut off' markers in --file, by running --run against "
            "every mutation."
        ),
    )
    parser.add_argument(
        "--file", required=True, help="source file containing unimut markers"
    )
    parser.add_argument(
        "--run",
        help=(
            "shell command to build/test each mutant; required unless "
            "--print-mutant-counts is given"
        ),
    )
    parser.add_argument(
        "--lang",
        choices=sorted(_LANGUAGES),
        default=None,
        help="override language detection from --file's extension",
    )
    parser.add_argument(
        "--print-mutant-counts",
        action="store_true",
        help="print how many mutants would be tried and exit, without running anything",
    )
    parser.add_argument(
        "--include-killed-mutants",
        action="store_true",
        help="also show mutants that were killed (--run failed), not just survivors",
    )
    parser.add_argument(
        "--diff",
        metavar="REF",
        default=None,
        help=(
            "PR-gate mode: scan the whole file but only mutate lines that "
            "differ from REF (a branch, tag, or commit), via "
            "'git diff REF...HEAD'. Implies whole-file scanning, so any "
            "// unimut on/off markers still present are read as "
            "exclusions, same as --whole-file."
        ),
    )
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help=(
            "nightly-audit mode: mutate the entire file instead of just "
            "marked regions. Any // unimut on/off pair still present "
            "is read inverted -- as a range to exclude, not include."
        ),
    )
    parser.add_argument(
        "--keep-call",
        metavar="NAME",
        action="append",
        default=None,
        dest="keep_call",
        help=(
            "never propose removing a statement that is just a call to "
            "NAME (e.g. --keep-call printf --keep-call assert), so "
            "logging/assertion calls don't get reported as untested; "
            "repeatable"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "run N mutants at a time, each in its own throwaway copy of "
            "the repository, for use on CI runners with spare cores "
            "(default: 1)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.jobs < 1:
        print("unimut: error: --jobs must be at least 1", file=sys.stderr)
        return 1

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"unimut: error: no such file: {file_path}", file=sys.stderr)
        return 1

    language = _detect_language(file_path, args.lang)
    original_source = file_path.read_text()

    # --diff implies whole-file scanning: it needs to see every statement
    # in the file to know which ones land on a changed line.
    whole_file = args.whole_file or args.diff is not None

    generate_params = inspect.signature(language.generate_mutants).parameters
    if whole_file and "whole_file" not in generate_params:
        print(
            f"unimut: error: the '{args.lang or file_path.suffix}' backend does "
            "not support --diff/--whole-file (marker-based regions only)",
            file=sys.stderr,
        )
        return 1

    changed_lines = None
    if args.diff is not None:
        changed_lines = _changed_lines(file_path, args.diff)
        if not changed_lines:
            print(
                f"unimut: no lines in {file_path} differ from '{args.diff}'",
                file=sys.stderr,
            )
            return 1

    if args.keep_call and "keep_calls" not in generate_params:
        print(
            f"unimut: error: the '{args.lang or file_path.suffix}' backend does "
            "not support --keep-call",
            file=sys.stderr,
        )
        return 1

    generate_kwargs = {}
    if whole_file:
        generate_kwargs["whole_file"] = True
        if changed_lines is not None:
            generate_kwargs["changed_lines"] = changed_lines
    if args.keep_call:
        generate_kwargs["keep_calls"] = set(args.keep_call)

    try:
        mutants = language.generate_mutants(
            str(file_path), original_source, **generate_kwargs
        )
    except mutate_c.MutationError as exc:
        print(f"unimut: error: {exc}", file=sys.stderr)
        return 1

    if not mutants:
        print(f"unimut: no mutable statements found in {file_path}", file=sys.stderr)
        return 1

    if args.print_mutant_counts:
        print(f"{len(mutants)} mutant{'s' if len(mutants) > 1 else ''}")
        return 0

    if not args.run:
        print(
            "unimut: error: --run is required unless --print-mutant-counts is given",
            file=sys.stderr,
        )
        return 1

    if shutil.which("sh") is None:
        print("unimut: error: no shell ('sh') found on PATH", file=sys.stderr)
        return 1

    repo_root = _git_repo_root(file_path.resolve().parent) or Path.cwd()
    baseline_survived, baseline_output, results = _run_mutants(
        file_path, original_source, mutants, args.run, args.jobs, repo_root
    )

    if not baseline_survived:
        print(
            "unimut: error: --run failed against the unmodified code -- fix "
            "your build/test command before running mutation tests "
            "(mutant results would be meaningless)",
            file=sys.stderr,
        )
        if baseline_output.strip():
            print(file=sys.stderr)
            print(baseline_output, file=sys.stderr, end="")
        return 1

    color = _use_color(sys.stdout)
    return _print_report(results, args.include_killed_mutants, color)


if __name__ == "__main__":
    sys.exit(main())
