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
``--keep-call printf --keep-call print_int`` stops it from reporting
your logging calls as "untested", since most applications never test
those in the first place. It's not meant for assertions: a statement
like ``assert(ptr != NULL);`` *should* still be offered as a mutant
(and fail your test once mutated to ``assert(ptr == NULL);``) -- that's
exactly the kind of code this tool exists to hold accountable.

``--timeout SECONDS`` (default 10) bounds how long any single ``--run``
invocation gets. A mutant that hangs past it (e.g. one that turned a
loop infinite) is killed and silently treated as killed, same as any
other non-surviving mutant; a baseline that times out is reported as an
error instead, same as any other baseline failure.

``--exit-on-first-survivor`` stops the whole run the moment any mutant
survives, rather than working through the rest. The full run still
tells you *everything* that's under-tested, but while iterating locally
you often just want to know "is there a gap at all" as fast as
possible; this skips (and, with ``--jobs`` > 1, cancels whatever's
already in flight for) every mutant after the first survivor.

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
import colorsys
import inspect
import itertools
import math
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
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


# ---------------------------------------------------------------------------
# Live progress bar: a spinner + "n/m survived" + an ETA, repainted on a
# fixed timer while mutants run, in the same wave-colored style as unimut's
# other animated bits. Purely cosmetic -- disabled outright when stdout
# isn't a terminal, so piping to a file or CI log doesn't fill up with
# carriage-return spam.
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_CLEAR_LINE = "\033[K"
_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"


def _wave_text(text: str, base_hue: float, t: float) -> str:
    """Color ``text`` character-by-character with a slowly shifting wave.

    Same technique start to finish as unimut's other animated output:
    each character's hue oscillates slightly around ``base_hue`` and its
    brightness dips and rises, as a function of ``t`` (seconds) and the
    character's position -- giving a gentle side-to-side shimmer as
    successive frames are rendered.
    """
    if not text:
        return text
    chars = []
    for i, ch in enumerate(text):
        wave = math.sin(t * 2.0 - i * 0.1)
        hue = (base_hue + wave * 0.03) % 1.0
        value = 1.0 - abs(wave * 0.3)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        chars.append(f"\033[38;2;{r};{g};{b}m{ch}")
    return "".join(chars) + "\033[0m"


def _format_eta(seconds: float) -> str:
    """``90s`` -> ``1m 30s``, ``5945s`` -> ``1h 39m 5s`` -- always ends in
    a bare (unpadded) seconds component, with any larger units it needs
    in front of it."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


class _Progress:
    """A live ``n/m run · k survived · <ETA|spinner>`` line, repainted
    ~10x/second.

    Examples of what gets rendered, as a run progresses::

        0/1002 run · 0 survived · ⠋              (no ETA yet: show the spinner)
        1/1002 run · 0 survived · 1h 39m 5s      (first result in: ETA takes over)
        1000/1002 run · 3 survived · 34s

    :meth:`record` is called from the main thread every time a mutant
    result comes in (see ``_run_mutants``'s ``as_completed`` loop) and
    just updates a couple of counters under a lock; a background thread
    does the actual (re)painting on a timer, independent of how bursty
    mutant completions are, so the spinner keeps animating smoothly even
    while waiting on a slow compile.
    """

    def __init__(self, total: int, stream=sys.stdout, base_hue: float = 0.55):
        self._total = total
        self._stream = stream
        self._base_hue = base_hue
        self._enabled = total > 0 and hasattr(stream, "isatty") and stream.isatty()
        self._lock = threading.Lock()
        self._completed = 0
        self._survived = 0
        self._start = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._stream.write(_CURSOR_HIDE)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def record(self, survived: bool) -> None:
        with self._lock:
            self._completed += 1
            if survived:
                self._survived += 1

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._stream.write(f"\r{_CLEAR_LINE}{_CURSOR_SHOW}")
        self._stream.flush()

    def _loop(self) -> None:
        spinner = itertools.cycle(_SPINNER_FRAMES)
        while not self._stop_event.is_set():
            self._paint(next(spinner))
            self._stop_event.wait(0.1)

    def _paint(self, frame: str) -> None:
        with self._lock:
            completed, survived = self._completed, self._survived
        if completed == 0:
            # No results yet to estimate a rate from -- show the spinner
            # instead of an ETA we can't actually compute.
            suffix = frame
        else:
            elapsed = time.time() - self._start
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = max(0, self._total - completed)
            eta_seconds = remaining / rate if rate > 0 else 0
            suffix = _format_eta(eta_seconds)
        message = f"{completed}/{self._total} run · {survived} survived · {suffix}"
        self._stream.write(
            f"\r{_wave_text(message, self._base_hue, time.time())}{_CLEAR_LINE}"
        )
        self._stream.flush()


def _kill_process_group(pid: int) -> None:
    """Best-effort kill of a process group. Fine if it's already gone."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_command(
    run_cmd: str, cwd: Optional[Path] = None, timeout: Optional[float] = None
) -> bool:
    """Run the user's --run command; True if it exited 0 (mutant survived).

    Runs in its own process group (``start_new_session=True``) so that on
    a timeout, the *whole* thing -- the shell plus whatever it spawned,
    e.g. a mutant that turned a loop infinite -- can be killed outright
    via that group, rather than ``subprocess``'s default timeout
    handling, which only touches the immediate shell process and would
    leave a runaway child behind. A timeout counts as "did not survive",
    silently, the same as any other non-zero exit.
    """
    proc = subprocess.Popen(
        run_cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid)
        proc.wait()
        return False
    return returncode == 0


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


def _worker_main(
    task_queue: "multiprocessing.Queue",
    result_queue: "multiprocessing.Queue",
    repo_root: str,
    rel_path: str,
    original_source: str,
    run_cmd: str,
    timeout_seconds: float,
) -> None:
    """Entry point run in each worker process.

    Ignores Ctrl-C outright (``SIGINT`` -> ignored) and puts itself in
    its own process group (``os.setpgrp()``): Ctrl-C is handled entirely
    by the parent process (see ``_run_mutants``), which reacts by
    killing each worker's whole process group in one shot -- taking down
    whatever build/test subprocess it's mid-way through along with it --
    rather than waiting for a graceful shutdown that a long-running
    compile would otherwise block on. Being its own process group leader
    means that kill lands on this worker and its subprocess without also
    hitting sibling workers, which are each leaders of their own,
    separate groups.

    Copies ``repo_root`` once into a fresh temp directory (reporting it
    back immediately, before doing anything else, so the parent can
    still clean it up even if this worker gets killed moments later),
    then processes ``("baseline",)`` and ``("mutant", idx, mutant)``
    tasks from ``task_queue`` until it sees the ``None`` sentinel.

    ``timeout_seconds`` (from ``--timeout``, already in seconds) bounds
    how long any single ``run_cmd`` invocation gets. A mutant that
    times out (e.g. one that turned a loop infinite) is silently treated
    as killed -- the same as any other non-surviving mutant, no fuss.
    The baseline is different: a baseline timeout is still a baseline
    *failure*, reported with an error the same as a baseline that fails
    to build, since either way no mutant result can be trusted.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.setpgrp()
    except (AttributeError, OSError):
        pass  # best-effort; e.g. unavailable on non-POSIX platforms

    workdir = Path(tempfile.mkdtemp(prefix="unimut-worker-"))
    shutil.copytree(
        repo_root, workdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git")
    )
    result_queue.put(("workdir", str(workdir)))
    target = workdir / rel_path

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            if task[0] == "baseline":
                # A worker's copy always matches original_source when it
                # picks this up (freshly copied, or restored after any
                # earlier mutant this same worker processed), so there's
                # nothing to write first. Output is captured, unlike
                # mutant runs, since a baseline failure -- including a
                # timeout -- is worth explaining to the person running
                # unimut. Its own process group (like _run_command) lets
                # a timeout kill the whole thing outright, not just the
                # immediate shell process.
                proc = subprocess.Popen(
                    run_cmd,
                    shell=True,
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, _ = proc.communicate(timeout=timeout_seconds)
                    result_queue.put(("baseline", proc.returncode == 0, stdout))
                except subprocess.TimeoutExpired:
                    _kill_process_group(proc.pid)
                    partial, _ = proc.communicate()
                    message = f"timed out after {timeout_seconds:g}s"
                    if partial and partial.strip():
                        message = f"{message}\n\n{partial}"
                    result_queue.put(("baseline", False, message))
            else:
                _, idx, mutant = task
                target.write_text(mutant.apply(original_source))
                try:
                    survived = _run_command(
                        run_cmd, cwd=workdir, timeout=timeout_seconds
                    )
                finally:
                    target.write_text(original_source)
                result_queue.put(("mutant", idx, survived))
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            result_queue.put(("error", repr(exc)))


def _run_mutants(
    file_path: Path,
    original_source: str,
    mutants: List,
    run_cmd: str,
    jobs: int,
    repo_root: Path,
    timeout_seconds: float,
    exit_on_first_survivor: bool = False,
) -> Tuple[bool, str, List[MutantResult], bool]:
    """Run a baseline check plus every mutant across ``jobs`` worker processes.

    Returns ``(baseline_survived, baseline_output, results, stopped_early)``.
    The baseline is a single extra unit of work -- "build/test the code with
    no mutation applied" -- sharing the same worker pool as the mutants,
    so it runs *alongside* them rather than serially in front of them.
    If it comes back failing, that means ``--run`` doesn't even pass
    against unmodified code, so no mutant result can be trusted; unimut
    kills every worker immediately and returns with ``results`` empty,
    rather than finishing a run whose conclusions would be meaningless.

    If ``exit_on_first_survivor`` is True, the moment any mutant comes
    back survived, unimut stops the same way: kills every worker
    immediately rather than waiting for whatever's still in flight, and
    returns whatever results had already come in (``stopped_early`` is
    True in this case, and only in this case -- a normal completed run,
    even one with survivors, reports False). This is for fast local
    iteration, where finding out about the *first* gap is enough to go
    fix something without waiting through however many mutants remain.

    ``timeout_seconds`` (from ``--timeout``) is passed straight through
    to each worker; see :func:`_worker_main` for what happens on a
    per-mutant vs. a baseline timeout.

    unimut never mutates ``--file`` (or anything else in the user's real
    working tree) in place. Each worker process gets its own full copy of
    ``repo_root`` in a temp directory (see :func:`_worker_main`), mutates
    *that* copy's version of the file, and runs ``run_cmd`` there -- so
    the original project is left untouched no matter how ``--run``
    behaves, and workers can't race each other writing to a shared file.

    This uses *processes*, not threads: applying a mutant re-parses the
    marked region with pycparser, which is CPU-bound pure Python and
    therefore holds the GIL, so threads would serialize on that step
    however many cores are idle. Separate processes don't share a GIL, so
    they actually scale with ``--jobs``.

    Workers are managed by hand with :mod:`multiprocessing` rather than
    ``concurrent.futures.ProcessPoolExecutor``: on Ctrl-C, that pool's
    graceful, atexit-driven shutdown has to wait for every in-flight
    build/test subprocess to finish on its own, which is exactly what
    you're trying to escape by hitting Ctrl-C, and in the meantime each
    worker's own default SIGINT handling raises a KeyboardInterrupt of
    its own, producing a flood of tracebacks. Here, workers ignore
    Ctrl-C entirely and the parent kills them outright the moment it
    sees one (see :func:`_kill_process_group`), so a single Ctrl-C is
    enough.
    """
    resolved_root = repo_root.resolve()
    rel_path = str(file_path.resolve().relative_to(resolved_root))

    ctx = multiprocessing.get_context("fork")
    task_queue: "multiprocessing.Queue" = ctx.Queue()
    result_queue: "multiprocessing.Queue" = ctx.Queue()

    task_queue.put(("baseline",))
    for idx, mutant in enumerate(mutants):
        task_queue.put(("mutant", idx, mutant))
    for _ in range(jobs):
        task_queue.put(None)

    workers = [
        ctx.Process(
            target=_worker_main,
            args=(
                task_queue,
                result_queue,
                str(resolved_root),
                rel_path,
                original_source,
                run_cmd,
                timeout_seconds,
            ),
            daemon=True,
        )
        for _ in range(jobs)
    ]
    for w in workers:
        w.start()

    results: List[Optional[MutantResult]] = [None] * len(mutants)
    workdirs: List[str] = []
    baseline_survived = True
    baseline_output = ""
    baseline_done = False
    completed_mutants = 0
    stopped_early = False

    def _cleanup(kill: bool) -> None:
        if kill:
            for w in workers:
                if w.pid is not None:
                    _kill_process_group(w.pid)
        for w in workers:
            w.join(timeout=5)
        # A queue with unflushed data can block interpreter shutdown on
        # its own background feeder thread; we're done with both, so
        # don't let that thread's join hold anything up.
        task_queue.cancel_join_thread()
        result_queue.cancel_join_thread()
        for wd in workdirs:
            shutil.rmtree(wd, ignore_errors=True)

    progress = _Progress(total=len(mutants))
    progress.start()
    try:
        while not baseline_done or completed_mutants < len(mutants):
            kind, *payload = result_queue.get()
            if kind == "workdir":
                workdirs.append(payload[0])
            elif kind == "error":
                raise RuntimeError(f"unimut worker failed: {payload[0]}")
            elif kind == "baseline":
                baseline_survived, baseline_output = payload
                baseline_done = True
                if not baseline_survived:
                    # No point finishing (or even starting) the rest of
                    # the mutants: their results would be meaningless if
                    # --run doesn't even pass unmodified.
                    break
            else:  # "mutant"
                idx, survived = payload
                results[idx] = MutantResult(mutants[idx], survived)
                progress.record(survived)
                completed_mutants += 1
                if survived and exit_on_first_survivor:
                    # Found what --exit-on-first-survivor is looking
                    # for -- no point running (or finishing) the rest.
                    stopped_early = True
                    break
    except BaseException:
        progress.stop()
        _cleanup(kill=True)
        raise
    else:
        progress.stop()
        _cleanup(kill=not baseline_survived or stopped_early)

    if not baseline_survived:
        return False, baseline_output, [], False
    return True, "", [r for r in results if r is not None], stopped_early


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
        "--exit-on-first-survivor",
        action="store_true",
        dest="exit_on_first_survivor",
        help=(
            "stop as soon as any mutant survives, instead of running the "
            "rest -- for fast local iteration, once you just want to "
            "know there's a gap rather than the full exhaustive count"
        ),
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
            "NAME (e.g. --keep-call printf --keep-call print_int), so "
            "logging calls don't get reported as untested; not meant "
            "for assertions, which should stay a mutation target; "
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help=(
            "kill any single --run invocation that takes longer than "
            "this many seconds (catches a mutation that e.g. turns a "
            "loop infinite). A mutant that times out is silently "
            "treated as killed, like any other non-surviving mutant; a "
            "baseline that times out is reported as an error, like any "
            "other baseline failure (default: 10 -- raise this if your "
            "own --run legitimately takes longer)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.jobs < 1:
        print("unimut: error: --jobs must be at least 1", file=sys.stderr)
        return 1

    if args.timeout <= 0:
        print(
            "unimut: error: --timeout must be greater than 0 (seconds)", file=sys.stderr
        )
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
    try:
        baseline_survived, baseline_output, results, stopped_early = _run_mutants(
            file_path,
            original_source,
            mutants,
            args.run,
            args.jobs,
            repo_root,
            args.timeout,
            args.exit_on_first_survivor,
        )
    except KeyboardInterrupt:
        print("\nunimut: cancelled", file=sys.stderr)
        return 130

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

    if stopped_early:
        skipped = len(mutants) - len(results)
        print(
            "unimut: a mutant survived -- stopping early "
            f"(--exit-on-first-survivor; {skipped} mutant"
            f"{'s' if skipped != 1 else ''} not run)",
            file=sys.stderr,
        )

    color = _use_color(sys.stdout)
    return _print_report(results, args.include_killed_mutants, color)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nunimut: cancelled", file=sys.stderr)
        sys.exit(130)
