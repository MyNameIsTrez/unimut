"""unimut: a small, focused mutation-testing CLI.

unimut looks inside ``// unimut start`` / ``// unimut stop`` markers in a
source file, generates every mutation of the code it finds there, and for
each mutation runs a user-supplied shell command (typically "build and run
the relevant tests"). Any mutant the command still passes against is a
*survivor*: a change your test suite failed to notice, which usually means
that code path is under-tested. Mutants that make the command fail --
whether because they broke compilation or because a test actually caught
the change -- are killed, and by default aren't shown at all.

Usage:

    unimut --file src/lj_ffrecord.c --run 'make -j$(nproc) && PATH="$PWD/src:$PATH" perl t/unpack.t'

Language support is dispatched by file extension (or ``--lang``) to a
separate backend module. Right now that's just ``mutate_c`` for C; see
``_LANGUAGES`` below for how to add more.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from . import mutate_c

# Maps a --lang value to the backend module that implements it. Each
# backend module must expose:
#   EXTENSIONS: set[str]                          -- recognized file extensions
#   generate_mutants(file_path, source) -> list[Mutant]
# where a Mutant has .file, .line, .original, .mutated (str | None) and
# an .apply(source) -> str method.
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


def _run_command(run_cmd: str) -> bool:
    """Run the user's --run command; True if it exited 0 (mutant survived)."""
    result = subprocess.run(
        run_cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _run_mutants(
    file_path: Path, original_source: str, mutants: List, run_cmd: str
) -> List[MutantResult]:
    results: List[MutantResult] = []
    with tempfile.TemporaryDirectory(prefix="unimut-") as tmpdir:
        backup_path = Path(tmpdir) / file_path.name
        backup_path.write_text(original_source)

        def restore():
            file_path.write_text(original_source)

        try:
            for mutant in mutants:
                mutated_source = mutant.apply(original_source)
                file_path.write_text(mutated_source)
                try:
                    survived = _run_command(run_cmd)
                finally:
                    restore()
                results.append(MutantResult(mutant, survived))
        except BaseException:
            # Make absolutely sure the user's real file is never left in a
            # mutated state, even on Ctrl-C or an unexpected error.
            restore()
            raise
    return results


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
            "Mutation-test the code between '// unimut start' and "
            "'// unimut stop' markers in --file, by running --run against "
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"unimut: error: no such file: {file_path}", file=sys.stderr)
        return 1

    language = _detect_language(file_path, args.lang)
    original_source = file_path.read_text()

    try:
        mutants = language.generate_mutants(str(file_path), original_source)
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

    results = _run_mutants(file_path, original_source, mutants, args.run)
    color = _use_color(sys.stdout)
    return _print_report(results, args.include_killed_mutants, color)


if __name__ == "__main__":
    sys.exit(main())
