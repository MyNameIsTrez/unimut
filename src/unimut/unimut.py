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

unimut never mutates ``--file`` in place. It always copies the whole
repository (as reported by ``git rev-parse --show-toplevel``, or the
current directory if that isn't a git checkout) into an isolated temp
directory, and mutates and tests that copy instead -- so the real
project on disk is never touched, and ``--jobs N`` (default 1) can run N
mutants at a time, each with its own throwaway copy, without workers
racing each other over a shared file.

Language support is dispatched by file extension (or ``--lang``) to a
separate backend module. Right now that's just ``mutate_c`` for C; see
``_LANGUAGES`` below for how to add more.
"""

from __future__ import annotations

import argparse
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Set

from . import mutate_c

# Maps a --lang value to the backend module that implements it. Each
# backend module must expose:
#   EXTENSIONS: set[str]                          -- recognized file extensions
#   generate_mutants(file_path, source, *, whole_file=False, changed_lines=None)
#       -> list[Mutant]
# where a Mutant has .file, .line, .original, .mutated (str | None) and
# an .apply(source) -> str method. The whole_file/changed_lines keyword
# arguments are optional -- a backend that only supports marker-based
# regions can omit them, and unimut will refuse --diff/--whole-file for
# that backend with a clear error rather than silently ignoring the flag.
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


def _run_mutants(
    file_path: Path,
    original_source: str,
    mutants: List,
    run_cmd: str,
    jobs: int,
    repo_root: Path,
) -> List[MutantResult]:
    """Run ``mutants`` across ``jobs`` worker threads (``jobs=1`` by default).

    unimut never mutates ``--file`` (or anything else in the user's real
    working tree) in place. Each worker gets its own full copy of
    ``repo_root`` in a temp directory, mutates *that* copy's version of
    the file, and runs ``run_cmd`` there -- so the original project is
    left untouched no matter how ``--run`` behaves, and, above ``--jobs
    1``, workers can't race each other writing to a shared file.
    """
    resolved_root = repo_root.resolve()
    rel_path = file_path.resolve().relative_to(resolved_root)

    results: List[Optional[MutantResult]] = [None] * len(mutants)
    work_queue = list(enumerate(mutants))
    queue_lock = threading.Lock()
    errors: List[BaseException] = []

    def worker() -> None:
        workdir = Path(tempfile.mkdtemp(prefix="unimut-worker-"))
        try:
            shutil.copytree(
                resolved_root,
                workdir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git"),
            )
            target = workdir / rel_path
            while not errors:
                with queue_lock:
                    if not work_queue:
                        return
                    idx, mutant = work_queue.pop()
                mutated_source = mutant.apply(original_source)
                target.write_text(mutated_source)
                try:
                    survived = _run_command(run_cmd, cwd=workdir)
                finally:
                    target.write_text(original_source)
                results[idx] = MutantResult(mutant, survived)
        except BaseException as exc:  # noqa: BLE001 - propagated to the caller
            errors.append(exc)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    threads = [threading.Thread(target=worker) for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
    return [r for r in results if r is not None]


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

    generate_kwargs = {}
    if whole_file:
        generate_kwargs["whole_file"] = True
        if changed_lines is not None:
            generate_kwargs["changed_lines"] = changed_lines

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
    results = _run_mutants(
        file_path, original_source, mutants, args.run, args.jobs, repo_root
    )

    color = _use_color(sys.stdout)
    return _print_report(results, args.include_killed_mutants, color)


if __name__ == "__main__":
    sys.exit(main())
