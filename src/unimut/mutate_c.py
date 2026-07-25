"""C mutation backend for unimut, built on pycparser.

This module knows how to:

  1. Find ``// unimut start`` / ``// unimut stop`` marked regions in a C
     source file.
  2. Parse those regions with pycparser and enumerate every *statement*
     that lives directly inside a ``{ ... }`` block (a pycparser
     ``Compound``'s ``block_items``).
  3. For each such statement, produce a ``Mutant`` whose ``apply()``
     regenerates the region with that one statement removed.

pycparser only understands strict C syntax: it has no C preprocessor and
no knowledge of any project's typedefs. Real-world snippets (like the
LuaJIT recorder code this tool was built for) reference types such as
``TRef`` or ``jit_State`` that pycparser has never heard of, and use
comments and calling-convention macros (``LJ_FASTCALL``) that pycparser's
grammar rejects outright.

To cope with that, before parsing we:

  * strip ``//`` and ``/* */`` comments (replacing them with blank space
    so line numbers do not shift),
  * strip bare ALL_CAPS tokens that sit directly in front of a function
    name (``RETTYPE MACRONAME name(...)``) since these are almost always
    calling-convention/attribute macros,
  * heuristically scan local declarations and function parameters for
    identifiers that look like unknown types, and inject
    ``typedef int TheirName;`` fake typedefs for them, plus a small fixed
    preamble of ``stdint.h``-style typedefs.

This is a best-effort heuristic, not a real preprocessor. It is good
enough to parse the statement *structure* of typical C (which is all a
"remove this statement" mutator needs), but it is not a general purpose
C frontend. If a region truly cannot be parsed this way, mutate_c raises
``MutationError`` explaining as much.

Because pycparser's generated code does not preserve original
formatting, applying a mutant regenerates the *entire* marked region
(not just the mutated statement) via pycparser's ``CGenerator``. Only
the marked region's text is touched -- everything outside
``// unimut start`` / ``// unimut stop`` is left byte-for-byte alone.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, List, Optional, Tuple

from pycparser import c_ast, c_generator, c_parser

MARK_START = "// unimut start"
MARK_STOP = "// unimut stop"

EXTENSIONS = {".c"}

# A small set of fake typedefs for the standard fixed-width integer types.
# pycparser has no knowledge of <stdint.h>, so without this preamble any
# snippet using int32_t, size_t, etc. would trip the same "unknown type"
# problem we work around for project-specific types below.
_STDINT_PREAMBLE = (
    "typedef unsigned char uint8_t;\n"
    "typedef signed char int8_t;\n"
    "typedef unsigned short uint16_t;\n"
    "typedef signed short int16_t;\n"
    "typedef unsigned int uint32_t;\n"
    "typedef signed int int32_t;\n"
    "typedef unsigned long uint64_t;\n"
    "typedef signed long int64_t;\n"
    "typedef unsigned long size_t;\n"
    "typedef long ptrdiff_t;\n"
    "typedef int bool;\n"
)

_KNOWN_BASE_TYPES = {
    "int", "char", "float", "double", "void", "short", "long", "unsigned",
    "signed", "_Bool", "bool", "size_t", "ptrdiff_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "struct", "union", "enum",
    "const", "static", "extern", "register", "volatile", "inline", "auto",
}
_C_KEYWORDS = _KNOWN_BASE_TYPES | {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "goto", "sizeof", "typedef", "restrict",
}

# ALL_CAPS token that sits directly in front of "identifier(" -- almost
# always a calling-convention or attribute macro (LJ_FASTCALL, __cdecl
# spelled in caps, etc). We drop these; they carry no structural meaning
# for statement-level mutation.
_CALLING_CONVENTION_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b(?=\s+\w+\s*\()")

# A local declaration at the start of a line: optional storage-class
# keywords, a type name, optional pointer stars, an identifier, then
# something that can only follow a declarator (=, ;, , or [).
_LOCAL_DECL_RE = re.compile(
    r"^\s*(?:static\s+|const\s+|register\s+|volatile\s+)*"
    r"([A-Za-z_]\w*)\s*\*{0,2}\s*[A-Za-z_]\w*\s*[=;,\[]"
)

# A single function parameter: optional const, a type name, optional
# pointer stars, an identifier, end of string.
_PARAM_RE = re.compile(r"^\s*(?:const\s+)?([A-Za-z_]\w*)\s*\*{0,2}\s*[A-Za-z_]\w*\s*$")

# A function header: name(params) {  -- used only to find parameter lists.
_FUNC_HEADER_RE = re.compile(r"\b\w+\s*\(([^()]*)\)\s*\{", re.MULTILINE)


class MutationError(Exception):
    """Raised when a region cannot be turned into mutants."""


@dataclasses.dataclass
class Region:
    """A single ``// unimut start`` / ``// unimut stop`` block."""

    start_line: int  # 1-indexed line number of the first line of code
    end_line: int  # 1-indexed line number of the last line of code
    code: str


@dataclasses.dataclass
class Mutant:
    """One candidate mutation.

    ``mutated`` is ``None`` for pure statement removal, since there is no
    replacement line to show in the diff-style report. Later mutation
    kinds (value tweaks, operator flips, ...) will populate it.
    """

    file: str
    line: int
    original: str
    mutated: Optional[str]
    _apply: Callable[[str], str] = dataclasses.field(repr=False, compare=False)

    def apply(self, source: str) -> str:
        """Return a full copy of ``source`` with this mutation applied."""
        return self._apply(source)


def find_regions(source: str) -> List[Region]:
    """Find all ``// unimut start`` / ``// unimut stop`` regions.

    Marker lines are matched by their stripped content, so indentation
    around them is fine. Regions must not be nested, and every start must
    have a matching stop.
    """
    lines = source.splitlines()
    regions: List[Region] = []
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
            code = "\n".join(lines[start_idx + 1 : i])
            regions.append(
                Region(start_line=start_idx + 2, end_line=i, code=code)
            )
            start_idx = None
    if start_idx is not None:
        raise MutationError(
            f"'{MARK_START}' marker at line {start_idx + 1} has no matching "
            f"'{MARK_STOP}'"
        )
    return regions


def _strip_calling_convention_macros(code: str) -> str:
    return _CALLING_CONVENTION_RE.sub("", code)


def _strip_comments_preserve_lines(code: str) -> str:
    """Remove // and /* */ comments while keeping line numbers intact."""
    out: List[str] = []
    i, n = 0, len(code)
    while i < n:
        if code[i : i + 2] == "/*":
            j = code.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("\n" * code.count("\n", i, j))
            i = j
        elif code[i : i + 2] == "//":
            j = code.find("\n", i)
            i = n if j == -1 else j
        else:
            out.append(code[i])
            i += 1
    return "".join(out)


def _guess_fake_typedefs(code: str):
    names = set()
    for line in code.splitlines():
        m = _LOCAL_DECL_RE.match(line)
        if m and m.group(1) not in _C_KEYWORDS:
            names.add(m.group(1))
    for hm in _FUNC_HEADER_RE.finditer(code):
        for part in hm.group(1).split(","):
            pm = _PARAM_RE.match(part)
            if pm and pm.group(1) not in _C_KEYWORDS:
                names.add(pm.group(1))
    return names


_WRAPPER_NAME = "__unimut_wrapper"


def _parse_region(code: str) -> Tuple[c_ast.FileAST, int, int, bool]:
    """Parse a region's code, returning
    ``(ast, preamble_line_count, n_fake_decls, wrapped)``.

    The AST's ``ext`` list starts with ``n_fake_decls`` synthetic typedef
    declarations that must be stripped back out before regenerating code.

    A marked region is either a whole function definition (as in the
    LuaJIT recorder functions unimut was built for), in which case it
    already parses at the top level and its body's statements are used
    directly, or a bare sequence of statements/declarations (as you'd
    write inside an existing function body), in which case it does *not*
    parse as valid top-level C on its own -- there is no such thing as a
    loose statement outside a function. For that case we wrap the region
    in a synthetic ``void __unimut_wrapper(void) { ... }`` just for
    parsing purposes, so pycparser sees the statements inside a
    ``Compound`` the same way a real function body would present them.
    ``wrapped`` tells the caller which of the two happened, since it
    changes how mutated code must be regenerated afterwards.
    """
    cleaned = _strip_calling_convention_macros(code)
    cleaned = _strip_comments_preserve_lines(cleaned)
    names = sorted(_guess_fake_typedefs(cleaned))
    fake_typedefs = "".join(f"typedef int {name};\n" for name in names)
    preamble = _STDINT_PREAMBLE + fake_typedefs
    # Every line in _STDINT_PREAMBLE is exactly one typedef declaration, so
    # this is also how many ast.ext slots the fixed preamble occupies.
    n_preamble_decls = _STDINT_PREAMBLE.count("\n") + len(names)

    direct_ast = None
    direct_error: Optional[Exception] = None
    try:
        direct_ast = c_parser.CParser().parse(
            preamble + cleaned, filename="<unimut-region>"
        )
    except Exception as exc:  # pycparser raises plain Exception/ParseError
        direct_error = exc

    has_funcdef = direct_ast is not None and any(
        isinstance(node, c_ast.FuncDef) for node in direct_ast.ext[n_preamble_decls:]
    )
    if direct_ast is not None and has_funcdef:
        return direct_ast, preamble.count("\n"), n_preamble_decls, False

    wrapped_code = f"void {_WRAPPER_NAME}(void) {{\n{cleaned}\n}}\n"
    try:
        wrapped_ast = c_parser.CParser().parse(
            preamble + wrapped_code, filename="<unimut-region>"
        )
    except Exception as exc:
        # Neither interpretation parsed; surface whichever error is more
        # likely to be useful (the direct one, if we have it).
        primary = direct_error or exc
        raise MutationError(
            "could not parse marked region as C -- unimut's C support is "
            "heuristic and cannot handle every construct; try narrowing the "
            f"// unimut start/stop markers ({primary})"
        ) from primary
    return wrapped_ast, preamble.count("\n") + 1, n_preamble_decls, True


def _children_by_path(node, path):
    for name, _child in node.children():
        yield name


def _find_block_item_paths(node, path: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    """Return the path of every statement inside a ``{ ... }`` block."""
    results: List[Tuple[str, ...]] = []
    for name, child in node.children():
        child_path = path + (name,)
        if "[" in name and name.split("[", 1)[0] == "block_items":
            results.append(child_path)
        results.extend(_find_block_item_paths(child, child_path))
    return results


def _get_by_path(root, path: Tuple[str, ...]):
    node = root
    for component in path:
        if "[" in component:
            attr, idx = component[:-1].split("[")
            node = getattr(node, attr)[int(idx)]
        else:
            node = getattr(node, component)
    return node


def _remove_by_path(root, path: Tuple[str, ...]) -> None:
    parent = _get_by_path(root, path[:-1])
    attr, idx = path[-1][:-1].split("[")
    del getattr(parent, attr)[int(idx)]


def _region_source(ast: c_ast.FileAST, n_fake_decls: int, wrapped: bool) -> str:
    """Regenerate C source for the real (non-typedef-preamble) content of ``ast``.

    We defer to pycparser's own ``CGenerator`` plumbing (``_generate_stmt``
    for statements inside a block, ``visit_FileAST`` for top-level
    declarations) rather than calling ``visit()`` on each node directly,
    since plain ``visit()`` does not add the trailing semicolons/newlines
    that only get added by the surrounding statement-list logic.
    """
    gen = c_generator.CGenerator()
    if wrapped:
        wrapper = ast.ext[n_fake_decls]
        items = wrapper.body.block_items or []
        return "".join(gen._generate_stmt(item) for item in items).rstrip("\n")
    region_ast = c_ast.FileAST(ast.ext[n_fake_decls:])
    return gen.visit(region_ast).rstrip("\n")


def _make_apply(region: Region, path: Tuple[str, ...]) -> Callable[[str], str]:
    def _apply(full_source: str) -> str:
        ast, _preamble_lines, n_fake_decls, wrapped = _parse_region(region.code)
        _remove_by_path(ast, path)
        new_region_code = _region_source(ast, n_fake_decls, wrapped)

        lines = full_source.splitlines()
        trailing_newline = full_source.endswith("\n")
        before = lines[: region.start_line - 1]
        after = lines[region.end_line :]
        result_lines = before + new_region_code.splitlines() + after
        result = "\n".join(result_lines)
        if trailing_newline:
            result += "\n"
        return result

    return _apply


def generate_mutants(file_path: str, source: str) -> List[Mutant]:
    """Generate every statement-removal mutant for ``source``.

    ``file_path`` is used only for the ``Mutant.file`` field shown in
    reports; the actual source text to mutate is ``source``.
    """
    mutants: List[Mutant] = []
    regions = find_regions(source)
    source_lines = source.splitlines()

    for region in regions:
        ast, preamble_lines, _n_fake_decls, _wrapped = _parse_region(region.code)
        for path in _find_block_item_paths(ast):
            node = _get_by_path(ast, path)
            coord_line = node.coord.line if node.coord else None
            if coord_line is None:
                continue
            region_local_line = coord_line - preamble_lines
            file_line = region.start_line + region_local_line - 1
            if 1 <= file_line <= len(source_lines):
                original_display = source_lines[file_line - 1].strip()
            else:
                # Should not normally happen, but fall back to the
                # regenerated text rather than crashing.
                region_code_lines = region.code.splitlines()
                idx = region_local_line - 1
                original_display = (
                    region_code_lines[idx].strip()
                    if 0 <= idx < len(region_code_lines)
                    else "<unknown>"
                )
            mutants.append(
                Mutant(
                    file=file_path,
                    line=file_line,
                    original=original_display,
                    mutated=None,
                    _apply=_make_apply(region, path),
                )
            )
    return mutants


# ---------------------------------------------------------------------------
# Self-tests. These compile hardcoded C strings with the system compiler to
# confirm mutate_c.py actually produces valid, correctly-mutated C -- no
# external fixture files needed. Run with:
#     python -m unittest unimut.mutate_c
# or simply:
#     python mutate_c.py
# ---------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import os

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

def _compiles(c_source: str) -> bool:
    """Compile a C source string with the system compiler; True on success."""
    assert _CC, "no C compiler found on PATH"
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "test.c")
        obj_path = os.path.join(tmp, "test.o")
        with open(src_path, "w") as f:
            f.write(c_source)
        result = subprocess.run(
            [_CC, "-c", src_path, "-o", obj_path],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


class TestFindRegions(unittest.TestCase):
    def test_single_region(self):
        src = textwrap.dedent(
            """\
            int before;
            // unimut start
            int a;
            int b;
            // unimut stop
            int after;
            """
        )
        regions = find_regions(src)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].start_line, 3)
        self.assertEqual(regions[0].end_line, 4)
        self.assertEqual(regions[0].code, "int a;\nint b;")

    def test_no_regions(self):
        self.assertEqual(find_regions("int x;\n"), [])

    def test_multiple_regions(self):
        src = textwrap.dedent(
            """\
            // unimut start
            int a;
            // unimut stop
            int mid;
            // unimut start
            int b;
            // unimut stop
            """
        )
        regions = find_regions(src)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].code, "int a;")
        self.assertEqual(regions[1].code, "int b;")

    def test_unmatched_start_raises(self):
        with self.assertRaises(MutationError):
            find_regions("// unimut start\nint a;\n")

    def test_unmatched_stop_raises(self):
        with self.assertRaises(MutationError):
            find_regions("int a;\n// unimut stop\n")

    def test_nested_start_raises(self):
        with self.assertRaises(MutationError):
            find_regions(
                "// unimut start\n// unimut start\nint a;\n// unimut stop\n"
            )


class TestSimpleRemoval(unittest.TestCase):
    """Two independent, side-effect-free statements: removing either one
    must still produce code that compiles."""

    SRC = textwrap.dedent(
        """\
        int add(int a, int b) {
          // unimut start
          int sum = a + b;
          int doubled = sum * 2;
          // unimut stop
          return sum;
        }
        """
    )

    def test_generates_one_mutant_per_statement(self):
        mutants = generate_mutants("add.c", self.SRC)
        self.assertEqual(len(mutants), 2)
        self.assertEqual(mutants[0].line, 3)
        self.assertEqual(mutants[0].original, "int sum = a + b;")
        self.assertIsNone(mutants[0].mutated)
        self.assertEqual(mutants[1].line, 4)
        self.assertEqual(mutants[1].original, "int doubled = sum * 2;")

    def test_removing_second_statement_compiles_and_removes_it(self):
        mutants = generate_mutants("add.c", self.SRC)
        mutant = mutants[1]  # "int doubled = sum * 2;"
        mutated_source = mutant.apply(self.SRC)
        self.assertNotIn("doubled", mutated_source)
        self.assertIn("int sum = a + b;", mutated_source)
        # Everything outside the markers must be untouched.
        self.assertIn("int add(int a, int b) {", mutated_source)
        self.assertIn("return sum;", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))

    def test_removing_first_statement_breaks_the_build(self):
        # Removing "int sum = ..." leaves "return sum;" referencing an
        # undeclared variable -- this mutant should fail to compile,
        # which is exactly the "ignored" case unimut relies on.
        mutants = generate_mutants("add.c", self.SRC)
        mutant = mutants[0]
        mutated_source = mutant.apply(self.SRC)
        self.assertNotIn("int sum = a + b;", mutated_source)
        if _CC:
            self.assertFalse(_compiles(mutated_source))

    def test_original_file_is_byte_identical_when_no_mutant_applied(self):
        # Sanity check that generate_mutants() never mutates its input.
        before = self.SRC
        generate_mutants("add.c", self.SRC)
        self.assertEqual(self.SRC, before)


class TestNestedBlocks(unittest.TestCase):
    SRC = textwrap.dedent(
        """\
        int classify(int n) {
          // unimut start
          int result = 0;
          if (n > 0) {
            result = 1;
          }
          // unimut stop
          return result;
        }
        """
    )

    def test_both_top_level_and_nested_statements_are_candidates(self):
        mutants = generate_mutants("classify.c", self.SRC)
        originals = [m.original for m in mutants]
        self.assertIn("int result = 0;", originals)
        self.assertIn("if (n > 0) {", originals)
        self.assertIn("result = 1;", originals)

    def test_removing_whole_if_compiles(self):
        mutants = generate_mutants("classify.c", self.SRC)
        mutant = next(m for m in mutants if m.original == "if (n > 0) {")
        mutated_source = mutant.apply(self.SRC)
        self.assertNotIn("result = 1;", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))


class TestRealWorldStyleSnippet(unittest.TestCase):
    """A trimmed-down stand-in for the kind of code unimut was built
    for: unknown project types, pointer params, a calling-convention
    macro, comments, and nested blocks -- but self-contained so it
    does not need real LuaJIT headers to compile."""

    SRC = textwrap.dedent(
        """\
        typedef struct jit_State jit_State;
        typedef struct RecordFFData RecordFFData;
        typedef int TRef;

        /* pretend recorder function */
        static void MY_FASTCALL recff_demo(jit_State *J, RecordFFData *rd)
        {
          // unimut start
          TRef tra = 1;  /* first ref */
          TRef trb = 2;  // second ref
          if (tra) {
            trb = tra;
          }
          // unimut stop
        }
        """
    )

    def test_parses_and_generates_mutants(self):
        mutants = generate_mutants("recff_demo.c", self.SRC)
        originals = [m.original for m in mutants]
        self.assertIn("TRef tra = 1;  /* first ref */", originals)
        self.assertIn("TRef trb = 2;  // second ref", originals)
        self.assertIn("if (tra) {", originals)

    def test_line_numbers_point_at_original_file(self):
        mutants = generate_mutants("recff_demo.c", self.SRC)
        by_original = {m.original: m.line for m in mutants}
        lines = self.SRC.splitlines()
        for original, line in by_original.items():
            self.assertEqual(lines[line - 1].strip(), original)


class TestUnparsableRegionRaises(unittest.TestCase):
    def test_garbage_region_raises_mutation_error(self):
        src = "// unimut start\nthis is not ) ( valid c at all {{{\n// unimut stop\n"
        with self.assertRaises(MutationError):
            generate_mutants("bad.c", src)

if __name__ == "__main__":
    unittest.main()
