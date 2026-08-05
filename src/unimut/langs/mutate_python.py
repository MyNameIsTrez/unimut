"""Python mutation backend for unimut, built on the stdlib ``ast`` module.

This is the minimal reference implementation of unimut's backend
protocol (see the "Adding another language" section in the README) --
deliberately simpler than ``mutate_c.py``, and deliberately incomplete:
it implements only the single most basic mutation kind ``mutate_c.py``
offers, whole-statement removal. Comparison-operator swaps, boundary
mutation, else-unwrapping, and everything else in ``mutate_c.py``'s list
are left for later (there's nothing Python-specific stopping any of
them; this file just hasn't grown them yet).

It's simpler than ``mutate_c.py`` for two independent reasons:

* Python has no preprocessor to work around. ``mutate_c.py``'s biggest
  chunk of code exists purely to cope with C code pycparser can't parse
  as-is -- unknown types, calling-convention macros, comments. None of
  that applies here: Python has no macros, and being dynamically typed,
  there's no such thing as an "unknown type" to guess the width of in
  the first place.

* This module parses the *entire file* in one ``ast.parse()`` call,
  rather than slicing out just the marked region's text and parsing
  that in isolation the way ``mutate_c.py`` does. Slicing wouldn't
  generally produce valid Python on its own: Python's indentation *is*
  syntax, so a region marked inside a method body carries indentation
  that's invalid as top-level code, unlike C's braces. Parsing the
  whole file sidesteps that entirely -- statements are found and
  removed from wherever they actually live in the real tree, and a
  mutant is produced by regenerating the *entire* file via
  ``ast.unparse()``.

That last point has one honest consequence worth knowing about:
``ast.unparse()`` only ever produces the mutated *copy* handed to
``--run`` -- it never touches what unimut's own report shows (the
``- ...`` line always comes straight from the real source text, the
same way ``mutate_c.py``'s ``_original_display`` works). But ``ast``
does not retain comments, so if you inspect a mutant's temp-directory
copy by hand, don't be surprised to find every comment gone from it --
harmless for actually running the code, since comments carry no
semantics, but a real difference from ``mutate_c.py``, which leaves
everything outside the mutated statement byte-for-byte untouched.

Only marker-based regions are supported: this module's
``generate_mutants`` takes no ``whole_file``/``changed_lines``/
``keep_calls`` keyword arguments, so unimut correctly refuses
``--diff``/``--whole-file``/``--keep-call`` for ``.py`` files with a
clear error rather than silently ignoring the flag.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import Iterator, List, Optional

from . import Mutant, MutationError

MARK_START = "# unimut on"
MARK_STOP = "# unimut off"

EXTENSIONS = {".py"}


@dataclasses.dataclass
class _Region:
    """A single ``# unimut on`` / ``# unimut off`` block, by line range only.

    Unlike ``mutate_c.py``'s ``Region``, this carries no isolated source
    text: statement discovery always works against the whole file's own
    AST (see the module docstring), so there's nothing to isolate.
    """

    start_line: int  # 1-indexed, first line of code after the marker
    end_line: int  # 1-indexed, last line of code before the marker


def find_regions(source: str) -> List[_Region]:
    """Find all ``# unimut on`` / ``# unimut off`` regions.

    Same marker-matching rules as ``mutate_c.py``'s ``find_regions``:
    matched by stripped line content (so indentation around a marker is
    fine), regions cannot nest, and every start needs a matching stop.
    Zero regions is not itself an error -- it just means
    :func:`generate_mutants` will return no mutants, the same as
    ``mutate_c.py`` when a file has no ``// unimut on``/``off`` pair.
    """
    lines = source.splitlines()
    regions: List[_Region] = []
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == MARK_START:
            if start_idx is not None:
                raise MutationError(
                    f"nested '{MARK_START}' marker at line {i + 1} "
                    f"(already inside a region opened at line {start_idx + 1})"
                )
            start_idx = i
        elif stripped == MARK_STOP:
            if start_idx is None:
                raise MutationError(
                    f"'{MARK_STOP}' marker at line {i + 1} has no matching "
                    f"'{MARK_START}'"
                )
            regions.append(_Region(start_line=start_idx + 2, end_line=i))
            start_idx = None
    if start_idx is not None:
        raise MutationError(
            f"'{MARK_START}' marker at line {start_idx + 1} has no matching "
            f"'{MARK_STOP}'"
        )
    return regions


def _iter_statement_lists(node: ast.AST) -> Iterator[List[ast.stmt]]:
    """Yield every ``list[ast.stmt]`` that hangs directly off ``node`` --
    a module/function/class's own ``.body``, plus the ``.orelse``/
    ``.finalbody`` an ``if``/``for``/``while``/``try`` can have -- then
    recurses into each statement found there in turn. ``try``'s
    ``except`` handlers carry a body one level deeper than any of those
    fields, so they get walked explicitly too.

    This is the direct analogue of ``mutate_c.py``'s walk over a
    ``Compound``'s ``block_items``: every place a single statement could
    be removed from without disturbing anything else.
    """
    for field in ("body", "orelse", "finalbody"):
        stmt_list = getattr(node, field, None)
        if (
            isinstance(stmt_list, list)
            and stmt_list
            and isinstance(stmt_list[0], ast.stmt)
        ):
            yield stmt_list
            for stmt in stmt_list:
                yield from _iter_statement_lists(stmt)
    for handler in getattr(node, "handlers", []):
        yield from _iter_statement_lists(handler)


class _RemoveStatementApply:
    """Picklable ``apply()``: remove one statement from the tree, then
    regenerate the whole file via ``ast.unparse()``.

    Statements are identified by ``(lineno, col_offset)`` rather than by
    holding a reference to the actual AST node: ``apply()`` re-parses
    ``source`` from scratch every time it's called, since a plain
    closure over a specific node object wouldn't survive the ``--jobs``
    pickle roundtrip to a worker process (the same constraint
    ``mutate_c.py``'s own ``_RemoveStatementApply`` documents) -- so
    there is no original tree left to hold a node reference into by the
    time ``apply()`` runs in the worker. A statement's starting
    position is otherwise sufficient to relocate it, since it's called
    with the exact same ``source`` text ``generate_mutants`` was given.
    """

    def __init__(self, lineno: int, col_offset: int):
        self._lineno = lineno
        self._col_offset = col_offset

    def __call__(self, source: str) -> str:
        tree = ast.parse(source)
        for stmt_list in _iter_statement_lists(tree):
            for i, stmt in enumerate(stmt_list):
                if stmt.lineno == self._lineno and stmt.col_offset == self._col_offset:
                    del stmt_list[i]
                    return ast.unparse(tree)
        raise MutationError(
            f"statement at {self._lineno}:{self._col_offset} not found on "
            "re-parse -- was apply() called with different source text "
            "than generate_mutants() saw?"
        )


def generate_mutants(file_path: str, source: str) -> List[Mutant]:
    """Generate one statement-removal mutant for every statement inside a
    ``# unimut on`` / ``# unimut off`` region.

    Deliberately minimal: see the module docstring for what's not
    implemented yet (every mutation kind beyond removal, and
    ``--diff``/``--whole-file``/``--keep-call``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MutationError(f"{file_path} is not valid Python: {exc}") from exc

    regions = find_regions(source)
    source_lines = source.splitlines()
    mutants: List[Mutant] = []
    for region in regions:
        for stmt_list in _iter_statement_lists(tree):
            for stmt in stmt_list:
                if not (region.start_line <= stmt.lineno <= region.end_line):
                    continue
                # Only the statement's own first physical line is shown --
                # unlike mutate_c.py, this doesn't reconstruct a fuller
                # preview for statements spanning several lines.
                original_display = source_lines[stmt.lineno - 1].strip()
                mutants.append(
                    Mutant(
                        file=file_path,
                        line=stmt.lineno,
                        original=original_display,
                        mutated=None,
                        _apply=_RemoveStatementApply(stmt.lineno, stmt.col_offset),
                    )
                )
    return mutants


# ---------------------------------------------------------------------------
# Self-tests. Hardcoded Python source strings, no fixture files -- same
# spirit as mutate_c.py's own self-tests, minus the need for a system C
# compiler (ast.parse() is the only "does this still parse" check needed).
# Run with:
#     python -m unittest unimut.langs.mutate_python
# or as part of the whole suite:
#     python -m unittest discover -s src -p "*.py" -v
# ---------------------------------------------------------------------------

import pickle
import textwrap
import unittest


class TestFindRegions(unittest.TestCase):
    def test_single_region(self):
        src = textwrap.dedent(
            """\
            def f():
                # unimut on
                x = 1
                # unimut off
                return x
            """
        )
        regions = find_regions(src)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].start_line, 3)
        self.assertEqual(regions[0].end_line, 3)

    def test_no_markers_is_not_an_error(self):
        self.assertEqual(find_regions("x = 1\n"), [])

    def test_nested_start_raises(self):
        src = "# unimut on\n# unimut on\n# unimut off\n"
        with self.assertRaises(MutationError):
            find_regions(src)

    def test_unmatched_stop_raises(self):
        with self.assertRaises(MutationError):
            find_regions("# unimut off\n")

    def test_unmatched_start_raises(self):
        with self.assertRaises(MutationError):
            find_regions("# unimut on\n")

    def test_multiple_regions(self):
        src = textwrap.dedent(
            """\
            # unimut on
            x = 1
            # unimut off
            y = 2
            # unimut on
            z = 3
            # unimut off
            """
        )
        regions = find_regions(src)
        self.assertEqual(len(regions), 2)


class TestBasicStatementRemoval(unittest.TestCase):
    SRC = textwrap.dedent(
        """\
        def demo(a, b, c):
            # unimut on
            a = a + 1
            b = b + 1
            c = c + 1
            # unimut off
            return a, b, c
        """
    )

    def test_one_mutant_per_statement(self):
        mutants = generate_mutants("demo.py", self.SRC)
        self.assertEqual(len(mutants), 3)
        self.assertEqual([m.line for m in mutants], [3, 4, 5])
        self.assertEqual(
            {m.original for m in mutants},
            {
                "a = a + 1",
                "b = b + 1",
                "c = c + 1",
            },
        )
        # Pure removal: no replacement line to show.
        self.assertTrue(all(m.mutated is None for m in mutants))

    def test_statement_outside_the_region_is_not_a_target(self):
        mutants = generate_mutants("demo.py", self.SRC)
        self.assertNotIn("return a, b, c", {m.original for m in mutants})

    def test_apply_removes_only_that_statement(self):
        mutants = generate_mutants("demo.py", self.SRC)
        removing_b = next(m for m in mutants if m.original == "b = b + 1")
        mutated_source = removing_b.apply(self.SRC)
        # ast.parse() succeeding is the Python analogue of mutate_c.py's
        # "if _CC: self.assertTrue(_compiles(...))" check.
        ast.parse(mutated_source)
        self.assertNotIn("b = b + 1", mutated_source)
        self.assertIn("a = a + 1", mutated_source)
        self.assertIn("c = c + 1", mutated_source)
        # ast.unparse() renders this as "return (a, b, c)" -- still the
        # same statement, just with parens ast.parse() didn't require.
        self.assertIn("a, b, c", mutated_source)


class TestNestedStatements(unittest.TestCase):
    """Statements nested inside if/for/try -- not just directly in a
    function body -- must be found too, the same way mutate_c.py's walk
    isn't limited to a function's own top-level block_items."""

    SRC = textwrap.dedent(
        """\
        def demo(items):
            total = 0
            # unimut on
            for item in items:
                if item > 0:
                    total = total + item
                else:
                    total = total - 1
            try:
                total = total + 1
            except ValueError:
                total = 0
            # unimut off
            return total
        """
    )

    def test_finds_statements_at_every_nesting_depth(self):
        mutants = generate_mutants("demo.py", self.SRC)
        originals = {m.original for m in mutants}
        self.assertIn("for item in items:", originals)
        self.assertIn("if item > 0:", originals)
        self.assertIn("total = total + item", originals)
        self.assertIn("total = total - 1", originals)
        self.assertIn("total = total + 1", originals)
        self.assertIn("total = 0", originals)
        # The whole try/except is itself a removable statement (its
        # header line is what's shown); the "except ValueError:" header
        # on its own is not, since an except clause is not an ast.stmt --
        # only the statements inside it (like "total = 0" above) are.
        self.assertIn("try:", originals)


class TestMutantIsPicklable(unittest.TestCase):
    """--jobs ships mutants to worker processes via pickle."""

    def test_roundtrip(self):
        src = textwrap.dedent(
            """\
            def f(x):
                # unimut on
                x = x + 1
                # unimut off
                return x
            """
        )
        mutant = generate_mutants("f.py", src)[0]
        roundtripped = pickle.loads(pickle.dumps(mutant))
        self.assertEqual(roundtripped.apply(src), mutant.apply(src))
        self.assertNotIn("x = x + 1", roundtripped.apply(src))


class TestInvalidSourceRaises(unittest.TestCase):
    def test_syntax_error_raises_mutation_error(self):
        with self.assertRaises(MutationError):
            generate_mutants("bad.py", "def f(:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
