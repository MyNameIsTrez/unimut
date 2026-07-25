# unimut

`unimut` (universal mutator) is a small, focused mutation-testing tool designed for local development to find the tests you are missing. It is called "universal" because it is built to work for many programming languages, and it allows you to register custom backends for any missing languages.

You temporarily mark a region of a source file with `// unimut start` / `// unimut stop` (these markers should not be committed) and pass `unimut` a command that builds and tests your project. It then tries every mutation of the marked code to see which ones your test suite fails to notice.

A mutant your tests catch (the `--run` command fails) is **killed** and, by
default, not shown -- that's the good outcome. A mutant your tests don't
catch (the `--run` command still exits 0) **survived**, and gets printed:
that's a sign you are missing a test.

```diff
$ unimut --file src/lj_ffrecord.c --run 'make -j$(nproc) && PATH="$PWD/src:$PATH" perl t/unpack.t'

src/lj_ffrecord.c:13
- if (tref_isnil(tri)) i = 1;

src/lj_ffrecord.c:27
- if (i > e) { rd->nres = 0; return; }

Survived: 2/35
```

Lines starting with `- ` (removed/original code) print in red, lines
starting with `+ ` (replacement code) print in green, when stdout is a
terminal.

## Installing

```
pip install unimut
```

or, from a checkout, for development:

```
pip install -e .
```

`unimut` requires Python 3.9+. The bundled C backend depends on
[`pycparser`](https://github.com/eliben/pycparser), which is installed
automatically.

## Marking a region

Wrap the code you want mutated in marker comments, on their own line:

```c
// unimut start
static void LJ_FASTCALL recff_unpack(jit_State *J, RecordFFData *rd)
{
  ...
}
// unimut stop
```

A region can be a whole function definition (as above), or just a run of
statements/declarations the way they'd appear inside a function body:

```c
void foo(void) {
  // unimut start
  int sum = a + b;
  int doubled = sum * 2;
  // unimut stop
  return sum;
}
```

A single file can contain multiple `start`/`stop` pairs; regions cannot be
nested.

## Running it

```
unimut --file <path> --run '<shell command>'
```

`--run` is any shell command; it's typically "rebuild, then run the
relevant tests" (`&&`-chained, as in the example above). `unimut`:

1. Reads `--file` and finds every `// unimut start` / `// unimut stop`
   region.
2. Dispatches to a language backend based on `--file`'s extension (`.c` ->
   the built-in C backend). Use `--lang c` to force the C backend
   regardless of extension -- handy when a `.c` file has been renamed to
   something else.
3. Backs up the original file content in a temporary directory.
4. For every mutant: writes the mutated file to `--file`, runs `--run`,
   records survived/killed, and restores the original file content before
   moving on to the next mutant. The original file is always restored, even
   if `--run` fails, unimut is interrupted, or an error occurs.
5. Prints every surviving mutant (plus killed ones too, with
   `--include-killed-mutants`) in the diff-like format shown above, then a
   final `Survived: <n>/<total>` line.

`unimut` exits `0` if no mutants survived, `1` if at least one did (or on
error) -- so it's usable as a CI gate.

### Options

| Flag | Meaning |
|---|---|
| `--file PATH` | source file containing unimut markers (required) |
| `--run CMD` | shell command to build/test each mutant (required unless `--print-mutant-counts`) |
| `--lang {c}` | override language detection from `--file`'s extension |
| `--print-mutant-counts` | print how many mutants would be tried, and exit, without running anything |
| `--include-killed-mutants` | also print killed mutants in the report, not just survivors |

### A note on "ignored" compile failures

If a mutation makes the code fail to compile, `--run` will simply fail
(the same as a genuine test failure), so that mutant is counted as killed
and, by default, not shown. There's no separate "ignored" bucket to
configure -- a mutant that doesn't compile is indistinguishable, from
`unimut`'s point of view, from one that compiles but gets caught by a
test, and *should* be indistinguishable: either way, nothing survived.

## The C backend (`mutate_c`)

The first version of the C backend can do one kind of mutation: **remove a
statement**. It walks every `{ ... }` block in the marked region and, for
each statement or declaration directly inside one, generates a mutant with
that single statement deleted. Nested blocks are recursed into, so an
`if (...) { ... }` can be removed as a whole *and* the statements inside it
are each separately removable too.

It's built on `pycparser`, which means it needs the marked region to
actually parse as C. `pycparser` has no C preprocessor and no idea what
your project's own types are called, so real-world snippets like the
LuaJIT recorder functions this tool was written for -- full of
project-specific types (`TRef`, `jit_State`, ...) and calling-convention
macros (`LJ_FASTCALL`) -- won't parse as-is. `mutate_c.py` works around
this with a best-effort, heuristic pre-pass before handing text to
`pycparser`:

* `//` and `/* */` comments are stripped (replaced with matching blank
  space, so line numbers used for reporting don't shift).
* A bare ALL_CAPS token sitting directly in front of `name(` is assumed to
  be a calling-convention/attribute macro and dropped.
* Local declarations and function parameters are scanned for identifiers
  that look like unknown types, and a `typedef int TheName;` stand-in is
  synthesized for each, alongside a small fixed preamble of
  `<stdint.h>`-style typedefs (`int32_t`, `size_t`, ...).

This is enough to recover the *statement structure* of typical C, which is
all a "remove this statement" mutator needs -- it is not a general-purpose
C frontend, and it will not correctly resolve types for anything more
elaborate (e.g. it treats every unknown type as if it were `int`-sized).
If a region genuinely can't be parsed this way, `unimut` reports a clear
error asking you to narrow the markers rather than silently doing the
wrong thing.

Because `pycparser`'s code generator doesn't preserve original formatting,
applying a mutant regenerates the *entire marked region* (not just the
mutated line) through `pycparser`'s `CGenerator`. Only text between
`// unimut start` and `// unimut stop` is ever touched -- everything
outside the markers is left byte-for-byte identical, and of course the
file is restored to its original content after every mutant is tried.

### Testing `mutate_c.py`

`mutate_c.py` carries its own test suite, written against hardcoded
multiline C strings baked directly into the file -- no external fixture
files needed. Tests that check whether mutated output is still valid C
compile it with whatever C compiler (`cc`, `gcc`, or `clang`) is found on
`PATH`; those checks are skipped (not failed) if none is available. Run
them with:

```
python -m unittest unimut.mutate_c -v
```

## Adding another language

Language backends are plain modules exposing:

```python
EXTENSIONS: set[str]  # e.g. {".c"}

def generate_mutants(file_path: str, source: str) -> list[Mutant]:
    ...
```

where a `Mutant` has `.file`, `.line`, `.original`, `.mutated` (`str` or
`None`) attributes and an `.apply(source: str) -> str` method that returns
a full mutated copy of `source`. Register the new module in `_LANGUAGES` in
`unimut.py`, keyed by the value you want accepted for `--lang`.

## Planned

* More C mutation kinds beyond statement removal (operator flips, constant
  tweaks, condition negation) -- these are the ones that will actually
  populate `Mutant.mutated` and print a `+` line.
* Additional language backends.
