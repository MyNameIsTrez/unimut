# unimut

`unimut` (universal mutator) is a [mutation testing](https://en.wikipedia.org/wiki/Mutation_testing) tool that finds tests you are missing, built to scale across a project's whole lifecycle: a fast, precise gate on individual PRs (`--diff`), an exhaustive nightly audit of legacy code (`--whole-file`), and parallel execution across CI cores (`--jobs`) -- all from one tool, working on any language with a registered backend.

`unimut` tries mutations of your code and runs a `--run` command (typically "rebuild, then test") against each one:

- A mutant your tests catch (`--run` fails) is **killed** -- the good outcome, and hidden by default.
- A mutant your tests miss (`--run` still exits 0) **survived** -- a sign you're missing a test -- and gets printed.

```diff
$ unimut --file src/lj_ffrecord.c --run 'make -j$(nproc) && PATH="$PWD/src:$PATH" perl t/unpack.t'

src/lj_ffrecord.c:13
- if (tref_isnil(tri)) i = 1;

src/lj_ffrecord.c:27
- if (i > e) { rd->nres = 0; return; }

Survived: 2/35
```

`- ` lines (removed code) print red; `+ ` lines (replacement code, for mutation kinds that have one) print green. `unimut` exits `0` if nothing survived, `1` otherwise (or on error) -- usable as a CI gate.

## Installing

```
pip install unimut
```

or `pip install -e .` from a checkout for development. Requires Python 3.9+; the bundled C backend pulls in [`pycparser`](https://github.com/eliben/pycparser) automatically.

## Choosing what to mutate

| Mode | What it mutates |
|---|---|
| Default | Code wrapped in `// unimut on` / `// unimut off` markers |
| `--diff REF` | Whole file, filtered to lines that differ from `REF` |
| `--whole-file` | Whole file, exhaustively |

### Default: marker-based

```c
// unimut on
static void LJ_FASTCALL recff_unpack(jit_State *J, RecordFFData *rd)
{
  ...
}
// unimut off
```

Wrap a whole function, or just a run of statements as they'd appear inside one. A file can hold multiple `on`/`off` pairs; they cannot be nested.

### `--diff REF`: a fast PR gate

```
unimut --file src/api.c --diff main --run 'make -j$(nproc) && make test'
```

Scans the whole file, but keeps only mutants on a line `git diff REF...HEAD -- src/api.c` reports as changed. A PR only has to prove the lines it touched are covered, not the whole file -- turning an hours-long whole-codebase run into a seconds-long diff-sized one. `REF` is anything `git diff` accepts (`main`, `origin/main`, a SHA); requires `--file` to be inside a git repo with `REF` resolvable.

`--diff` implies whole-file scanning, so the marker inversion below applies to it too.

### `--whole-file`: a slow nightly audit

```
unimut --file src/api.c --whole-file --run 'make -j$(nproc) && make test'
```

Mutates every statement in the file -- the right mode for periodically auditing legacy code that never got markers. If `// unimut on`/`off` markers *are* still present, their meaning **inverts**: an `off`/`on` pair now marks a range to *exclude*, the same way tools like clang-format reuse on/off markers:

```c
// unimut off
die("Out of memory"); // no test can reliably trigger this allocator failure
// unimut on
```

Excluded text still has to be part of a file that parses as C overall -- exclusion hides a range from *mutation*, not from parsing.

## Running it

```
unimut --file <path> --run '<shell command>' [--diff REF | --whole-file] [--jobs N] [--keep-call NAME ...]
```

`unimut` never mutates your real files. It copies the whole repository (via `git rev-parse --show-toplevel`, or the current directory if that isn't a git checkout) into an isolated temp directory per job, and mutates and builds/tests that copy instead -- your working tree is untouched even if `--run` crashes or you hit Ctrl-C.

`--jobs N` runs N mutants at a time, each in its own worker process with its own repo copy -- not threads. Applying a mutant re-parses code with `pycparser`, which is CPU-bound pure Python and holds the GIL, so threads would mostly serialize on that step regardless of idle cores; separate processes don't share a GIL and actually scale with `--jobs`.

Every run also does one baseline check: build/test the code completely unmodified, to confirm `--run` actually passes before trusting any mutant result. It's not run up front and serially -- it's just one more job in the same worker pool, so it costs no extra wall time when it passes. If it fails -- including timing out, see below -- mutant results would be meaningless (a broken build "survives" every mutation), so unimut cancels whatever mutants haven't started yet and reports the baseline's own output instead of the usual survivor list.

`--timeout SECONDS` (default 10) bounds how long any single `--run` invocation gets, to catch mutations that hang -- e.g. removing a loop's increment and turning it infinite. A mutant that times out is killed (the whole process tree, not just the immediate shell) and silently treated as killed, same as any other non-surviving mutant. Raise it if your own `--run` legitimately takes longer than that (a slow test suite, a heavy build) -- and lower it if you want faster feedback on infinite-loop-style mutants and know your real runs are quick.

While mutants run, a live `n/m survived · ETA` line updates in place (spinner included) if stdout is a terminal; piping to a file or CI log disables it and prints nothing extra.

### Options

| Flag | Meaning |
|---|---|
| `--file PATH` | source file to mutate (required) |
| `--run CMD` | shell command to build/test each mutant (required unless `--print-mutant-counts`) |
| `--lang {c}` | override language detection from `--file`'s extension |
| `--diff REF` | PR-gate mode (see above) |
| `--whole-file` | nightly-audit mode (see above) |
| `--jobs N` | run N mutants at a time, each in its own isolated process (default: 1) |
| `--timeout SECONDS` | kill (and silently treat as killed) any mutant whose `--run` exceeds this many seconds; a baseline timeout is an error instead (default: 10) |
| `--keep-call NAME` | never remove a statement that's just a call to `NAME` (e.g. `--keep-call printf --keep-call assert`); repeatable |
| `--print-mutant-counts` | print how many mutants would be tried, and exit |
| `--include-killed-mutants` | also print killed mutants, not just survivors |

### A note on "ignored" compile failures

A mutation that fails to compile just makes `--run` fail like any other test failure, so it's counted and treated as killed -- there's no separate "ignored" bucket. That's intentional: a mutant that doesn't compile is indistinguishable from one a test caught, and *should* be, since either way nothing survived.

## The C backend (`mutate_c`)

The C backend does one kind of mutation so far: **remove a statement**. It walks every `{ ... }` block and generates one mutant per statement/declaration removed, recursing into nested blocks.

It's built on `pycparser`, which has no preprocessor and no idea what your project's types are called. Real code (like the LuaJIT recorder functions this was built for) uses unknown types (`TRef`, `jit_State`) and calling-convention macros (`LJ_FASTCALL`) that won't parse as-is, so `mutate_c.py` does a heuristic pre-pass first: strip comments (preserving line numbers), drop bare ALL_CAPS tokens in front of `name(`, and synthesize fake `typedef int Name;` stand-ins for identifiers that look like unknown types, on top of a small `<stdint.h>`-style preamble.

This recovers the *statement structure* of typical C -- enough for statement removal -- but it's not a general C frontend, and treats every unknown type as `int`-sized. A region that genuinely can't be parsed this way raises a clear error rather than silently doing the wrong thing.

Because `pycparser`'s generator doesn't preserve formatting, applying a mutant regenerates the whole marked (or whole-file) region through `pycparser`'s `CGenerator`; everything outside it is left byte-for-byte identical.

Run its own test suite (hardcoded C strings, no fixture files, compiled with whatever of `cc`/`gcc`/`clang` is on `PATH`) with:

```
python -m unittest discover -s src -p "*.py" -v
```

## Adding another language

Language backends are plain modules exposing:

```python
EXTENSIONS: set[str]  # e.g. {".c"}

def generate_mutants(file_path: str, source: str) -> list[Mutant]:
    ...
```

where `Mutant` has `.file`, `.line`, `.original`, `.mutated` (`str | None`), and an `.apply(source: str) -> str` method that must be picklable (`--jobs` ships mutants to worker processes -- a plain closure won't survive that; see `_RemoveStatementApply` in `mutate_c.py` for the pattern). To support `--diff`/`--whole-file`, also accept `whole_file: bool = False` and `changed_lines: set[int] | None = None`; to support `--keep-call`, accept `keep_calls: set[str] | None = None`. Backends that omit any of these simply refuse the corresponding flag. Register the module in `_LANGUAGES` in `unimut.py`, keyed by the `--lang` value.

## Planned

* More C mutation kinds beyond statement removal (operator flips, constant tweaks, condition negation) -- these will populate `Mutant.mutated` and print a `+` line.
* Additional language backends.

## Contributing

This project uses Black and Pyright. Run once to install a pre-commit hook that formats/checks staged files on every `git commit`:

```sh
pip install pre-commit && pre-commit install
```
