"""C mutation backend for unimut, built on pycparser.

This module knows how to:

  1. Find ``// unimut on`` / ``// unimut off`` marked regions in a C
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
``// unimut on`` / ``// unimut off`` is left byte-for-byte alone.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

from pycparser import c_ast, c_generator, c_parser

MARK_START = "// unimut on"
MARK_STOP = "// unimut off"

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
    "int",
    "char",
    "float",
    "double",
    "void",
    "short",
    "long",
    "unsigned",
    "signed",
    "_Bool",
    "bool",
    "size_t",
    "ptrdiff_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "struct",
    "union",
    "enum",
    "const",
    "static",
    "extern",
    "register",
    "volatile",
    "inline",
    "auto",
}
_C_KEYWORDS = _KNOWN_BASE_TYPES | {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "goto",
    "sizeof",
    "typedef",
    "restrict",
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
    """A single ``// unimut on`` / ``// unimut off`` block."""

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
    """Find all ``// unimut on`` / ``// unimut off`` regions.

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
            regions.append(Region(start_line=start_idx + 2, end_line=i, code=code))
            start_idx = None
    if start_idx is not None:
        raise MutationError(
            f"'{MARK_START}' marker at line {start_idx + 1} has no matching "
            f"'{MARK_STOP}'"
        )
    return regions


def whole_file_region(source: str) -> Region:
    """Treat the entire file as a single region, for ``--whole-file``/``--diff``.

    Used instead of :func:`find_regions` when unimut is auditing a whole
    file rather than an explicitly marked snippet. Marker comments (if any
    are still present in the file) are left in place here -- they get
    stripped out later as ordinary ``//`` comments during parsing, and any
    line ranges they bracket are excluded separately by
    :func:`find_excluded_ranges`.
    """
    lines = source.splitlines()
    return Region(start_line=1, end_line=len(lines), code=source)


def find_excluded_ranges(source: str) -> List[Tuple[int, int]]:
    """Find marker-bracketed ranges to *exclude* from a whole-file audit.

    In ``--whole-file``/``--diff`` mode the markers' meaning inverts: by
    default every statement in the file is a mutation candidate, and a
    ``// unimut off`` / ``// unimut on`` pair (in that order) instead
    carves out a range that should be *skipped* -- the same markers
    developers already use to mark an included region in the normal
    marker-based mode, just read the other way around. This lets you keep
    a handful of "no test can reliably trigger this" lines out of a
    nightly audit without polluting the rest of the file.

    Returns a list of inclusive ``(start_line, end_line)`` line ranges
    (1-indexed, covering the marker comment lines themselves as well as
    everything between them). Raises ``MutationError`` on an unmatched
    marker, mirroring the errors :func:`find_regions` raises for the
    non-inverted case.
    """
    lines = source.splitlines()
    excluded: List[Tuple[int, int]] = []
    stop_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == MARK_STOP:
            if stop_idx is not None:
                raise MutationError(
                    f"a second '{MARK_STOP}' marker at line {i + 1} was found "
                    f"before the exclusion opened at line {stop_idx + 1} was "
                    f"closed with '{MARK_START}'"
                )
            stop_idx = i
        elif stripped == MARK_START:
            if stop_idx is None:
                raise MutationError(
                    f"'{MARK_START}' marker at line {i + 1} has no preceding "
                    f"'{MARK_STOP}' to close -- in whole-file mode, markers "
                    f"mark an *excluded* range and must appear as "
                    f"'{MARK_STOP}' first, then '{MARK_START}'"
                )
            excluded.append((stop_idx + 1, i + 1))
            stop_idx = None
    if stop_idx is not None:
        raise MutationError(
            f"'{MARK_STOP}' marker at line {stop_idx + 1} has no matching "
            f"'{MARK_START}' to close the excluded range"
        )
    return excluded


def _line_excluded(line: int, excluded_ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in excluded_ranges)


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
            f"// unimut on/off markers ({primary})"
        ) from primary
    return wrapped_ast, preamble.count("\n") + 1, n_preamble_decls, True


def _children_by_path(node, path):
    for name, _child in node.children():
        yield name


# Attribute names (as returned by pycparser's ``Node.children()``) that hold
# a *mandatory* single child statement -- the "then" branch of an ``if`` and
# the body of a ``while``/``do``/``for`` loop. C requires *some* statement
# there, so when one of these isn't a ``{ ... }`` block it's just a bare
# statement hanging directly off the ``If``/``While``/``DoWhile``/``For``
# node -- never an entry in anyone's ``block_items`` list. Without treating
# the attribute itself as a candidate, a one-liner like ``if (cond) return;``
# would only ever offer "remove the whole if" as a mutant, never "remove
# just the return and leave the condition being evaluated for nothing" --
# a meaningfully different (and often uncaught) mutant.
_MANDATORY_BODY_ATTRS = {"iftrue", "stmt"}

# Attribute names that hold an *optional* single child: only ``If.iffalse``
# (the ``else`` clause) today. Whether or not it's braced, "delete the whole
# else branch" is a distinct, worthwhile mutant that -- like the mandatory
# bodies above -- never shows up as a ``block_items`` entry anywhere, so it
# needs the same explicit treatment.
_OPTIONAL_BODY_ATTRS = {"iffalse"}

# A synthetic path component (never a real pycparser attribute name) marking
# a third kind of ``if``/``else`` mutant: replace the *entire* ``if``
# statement -- condition, ``iftrue`` branch and all -- with just its
# ``else`` clause's content, executed unconditionally. This is different
# from both removing the whole ``if`` (which drops the else branch too) and
# removing just the else clause (which keeps the ``if`` guarding the then
# branch): here the then branch and the condition both disappear, and
# whatever the else clause held becomes the new, unconditional code. It's
# appended as an extra path alongside wherever the ``If`` node itself was
# found, so it's always paired with a plain deletion candidate for the same
# node.
_UNWRAP_ELSE_MARKER = "__unwrap_else__"


def _find_block_item_paths(node, path: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    """Return the path of every statement-removal candidate reachable from
    ``node``: every entry of every ``{ ... }`` block's ``block_items``, plus
    every bare (unbraced) ``if``/loop body, every ``else`` clause, and every
    "collapse to the else branch" candidate (see ``_MANDATORY_BODY_ATTRS``/
    ``_OPTIONAL_BODY_ATTRS``/``_UNWRAP_ELSE_MARKER`` above)."""
    results: List[Tuple[str, ...]] = []
    for name, child in node.children():
        child_path = path + (name,)
        if "[" in name and name.split("[", 1)[0] == "block_items":
            results.append(child_path)
        elif name in _OPTIONAL_BODY_ATTRS:
            results.append(child_path)
        elif name in _MANDATORY_BODY_ATTRS and not isinstance(child, c_ast.Compound):
            results.append(child_path)
        if isinstance(child, c_ast.If) and child.iffalse is not None:
            results.append(child_path + (_UNWRAP_ELSE_MARKER,))
        results.extend(_find_block_item_paths(child, child_path))
    return results


def _get_by_path(root: c_ast.Node, path: Tuple[str, ...]) -> c_ast.Node:
    node: c_ast.Node = root
    for component in path:
        if component == _UNWRAP_ELSE_MARKER:
            # Purely a marker for "this If, but the unwrap-else mutation" --
            # it doesn't navigate anywhere, the node of interest is the If
            # itself, already reached by the preceding path components.
            continue
        elif "[" in component:
            attr, idx = component[:-1].split("[")
            node = getattr(node, attr)[int(idx)]
        else:
            node = getattr(node, component)
    return node


def _node_line(node: c_ast.Node) -> Optional[int]:
    """``node.coord.line``, or None if ``node`` has no coordinate info."""
    coord = node.coord
    return coord.line if coord else None


def _unwrap_else(root: c_ast.Node, if_path: Tuple[str, ...]) -> None:
    """Replace the ``If`` node at ``if_path`` with its own ``iffalse``
    clause, wherever that ``If`` itself lives -- a ``block_items`` list
    slot, or a bare ``iftrue``/``stmt`` attribute of some further-out
    construct. Either way, this is a plain substitution: the condition and
    the ``iftrue`` branch vanish, and the else clause's content takes the
    ``If``'s place verbatim, now unconditional.
    """
    if_node = _get_by_path(root, if_path)
    assert isinstance(if_node, c_ast.If), f"unwrap-else path {if_path!r} is not an If"
    iffalse = if_node.iffalse
    assert iffalse is not None
    container = _get_by_path(root, if_path[:-1])
    last = if_path[-1]
    if "[" in last:
        attr, idx = last[:-1].split("[")
        getattr(container, attr)[int(idx)] = iffalse
    else:
        setattr(container, last, iffalse)


def _remove_by_path(root, path: Tuple[str, ...]) -> None:
    if path[-1] == _UNWRAP_ELSE_MARKER:
        _unwrap_else(root, path[:-1])
        return
    parent = _get_by_path(root, path[:-1])
    last = path[-1]
    if "[" in last:
        attr, idx = last[:-1].split("[")
        del getattr(parent, attr)[int(idx)]
    elif last in _OPTIONAL_BODY_ATTRS:
        # else clause: simply absent, same as if it was never written.
        setattr(parent, last, None)
    else:
        assert last in _MANDATORY_BODY_ATTRS, f"unrecognized removal path {last!r}"
        # if/loop body: C has no "no statement here" for these positions,
        # so the closest thing to "removed" is the empty statement -- the
        # condition (still) gets evaluated, but nothing happens as a result.
        setattr(parent, last, c_ast.EmptyStatement(coord=getattr(parent, last).coord))


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


@dataclasses.dataclass
class _RemoveStatementApply:
    """Picklable ``Mutant.apply()`` implementation for statement removal.

    This used to be a closure returned by a ``_make_apply(region, path)``
    factory function, which worked fine as long as everything ran in one
    process. ``--jobs`` runs mutants in worker *processes*, though, and
    Python's ``pickle`` (which ``multiprocessing`` uses to ship work to
    those processes) cannot serialize closures over local variables --
    only module-level classes and their (picklable) field values. Hence a
    small dataclass instead: same behavior, but it survives being pickled.
    """

    region: Region
    path: Tuple[str, ...]

    def __call__(self, full_source: str) -> str:
        ast, _preamble_lines, n_fake_decls, wrapped = _parse_region(self.region.code)
        _remove_by_path(ast, self.path)
        new_region_code = _region_source(ast, n_fake_decls, wrapped)

        lines = full_source.splitlines()
        trailing_newline = full_source.endswith("\n")
        before = lines[: self.region.start_line - 1]
        after = lines[self.region.end_line :]
        result_lines = before + new_region_code.splitlines() + after
        result = "\n".join(result_lines)
        if trailing_newline:
            result += "\n"
        return result


def _make_apply(region: Region, path: Tuple[str, ...]) -> Callable[[str], str]:
    return _RemoveStatementApply(region, path)


def _unwrap_cast(node):
    """Peel off Cast wrappers, e.g. ``(void)printf(...)`` -> the FuncCall."""
    while isinstance(node, c_ast.Cast):
        node = node.expr
    return node


def _call_name(node) -> Optional[str]:
    """If ``node`` *is* (possibly cast) a call to a plain-named function,
    return that function's name; otherwise None.

    Used by ``--keep-call``/``keep_calls`` to recognize statements like
    ``printf("%d\\n", 1 + 2);`` or ``assert(x > 0);`` that are nothing
    but a single call -- as opposed to a call buried inside a bigger
    statement, which statement removal would delete along with
    everything else around it anyway.
    """
    node = _unwrap_cast(node)
    if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
        return node.name.name
    return None


def _is_ancestor_path(prefix: Tuple[str, ...], path: Tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _find_line_unit_path(
    all_paths: List[Tuple[str, ...]],
    line_of: Dict[Tuple[str, ...], Optional[int]],
    target_path: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Return the shallowest registered block-item path that is
    ``target_path`` itself or one of its ancestors and shares its source
    line -- i.e. the outermost statement whose regenerated text corresponds
    to the *entire* original physical line containing the mutation.

    For a plain one-statement-per-line removal this is just ``target_path``
    itself. For something nested inside a one-line compound (the ``rd->nres
    = 0;`` inside ``if (i > e) { rd->nres = 0; return; }``), it's the
    enclosing ``if`` -- regenerating just the nested statement wouldn't
    reproduce the ``if (...) { ... }`` shell the removed statement lived in.
    """
    target_line = line_of.get(target_path)
    candidates = [
        p
        for p in all_paths
        if line_of.get(p) == target_line and _is_ancestor_path(p, target_path)
    ]
    return min(candidates, key=len)


def _decl_merge_signature(decl: c_ast.Decl):
    """Return a hashable signature describing everything about ``decl``
    except its declared name, or ``None`` if its structure is too complex to
    safely fold back into a merged ``TYPE a, b;`` declarator list (a pointer
    or array declarator, an initializer, a bitfield, or a struct/union/enum
    type all make the per-declarator text diverge, so those are left to the
    generic "render each survivor on its own" fallback instead).
    """
    if (
        decl.init is not None
        or decl.bitsize is not None
        or not isinstance(decl.type, c_ast.TypeDecl)
        or not isinstance(decl.type.type, c_ast.IdentifierType)
    ):
        return None
    return (
        tuple(decl.funcspec or ()),
        tuple(decl.storage or ()),
        tuple(decl.quals or ()),
        tuple(decl.type.quals or ()),
        tuple(decl.type.type.names),
    )


def _merge_decl_group(decls: List[c_ast.Decl]) -> Optional[str]:
    """Render the surviving declarators of a multi-declarator statement
    (e.g. the ``e, k`` left after removing ``i`` from ``int32_t i, e,
    k;``) back into one combined declaration, or None if that's not safe
    (see :func:`_decl_merge_signature`) -- in which case the caller falls
    back to rendering each survivor as its own statement.
    """
    if not decls:
        return None
    sig = _decl_merge_signature(decls[0])
    if sig is None or any(_decl_merge_signature(d) != sig for d in decls[1:]):
        return None
    funcspec, storage, decl_quals, type_quals, names = sig
    prefix = "".join(s + " " for s in funcspec) + "".join(s + " " for s in storage)
    prefix += "".join(q + " " for q in decl_quals)
    type_str = "".join(q + " " for q in type_quals) + " ".join(names)
    declarators = ", ".join(d.name for d in decls)
    return f"{prefix}{type_str} {declarators};"


def _collapse_to_one_line(text: str) -> str:
    """Collapse pycparser's pretty-printed (possibly multi-line, indented)
    rendering of a statement back into the single physical line it
    originated from, turning e.g.::

        if (i > e)
        {
          return;
        }

    into ``if (i > e) { return; }``, matching the style of the one-liner it
    replaces in the report.
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _mandatory_body_replacement(
    gen: c_generator.CGenerator,
    ast: c_ast.FileAST,
    target_path: Tuple[str, ...],
    line_of: Dict[Tuple[str, ...], Optional[int]],
) -> str:
    """Build the ``+`` preview text for removing an ``if``/loop body
    (an ``iftrue``/``stmt`` candidate from ``_MANDATORY_BODY_ATTRS``).

    This is deliberately its own thing rather than a call into
    :func:`_compute_mutated_text`'s general ancestor-collapse logic below,
    which assumes the "unit" it regenerates corresponds to exactly one
    physical source line. That assumption holds for the nested-statement
    case it was built for (a statement inside a single-line ``{ ... }``
    compound), but not here: an ``If`` node's own coordinate can share the
    removed body's line while its ``else`` clause lives several lines
    later, and naively regenerating the whole ``If`` (as the general path
    would) drags that unrelated, unmutated ``else`` block into the ``+``
    line. The replacement here is always scoped to exactly the physical
    line the removed body itself occupied.
    """
    parent = _get_by_path(ast, target_path[:-1])
    body_line = line_of[target_path]
    header_line = _node_line(parent)

    if header_line != body_line:
        # The body lives on its own line, separate from the
        # if/while/for/do header (e.g. ``if (cond)\n  stmt;``) -- removing
        # it just turns that one line into an empty statement; the header
        # is on a different, unaffected line (and reported separately, if
        # it's itself a candidate).
        return ";"

    if isinstance(parent, c_ast.If):
        cond = gen.visit(parent.cond) if parent.cond is not None else ""
        text = f"if ({cond}) ;"
        if parent.iffalse is not None and _node_line(parent.iffalse) == body_line:
            # The else clause is crammed onto this very same physical
            # line -- keep it, unchanged, in the reconstructed line.
            else_text = _collapse_to_one_line(gen._generate_stmt(parent.iffalse))
            text += f" else {else_text}"
        return text

    if isinstance(parent, c_ast.While):
        cond = gen.visit(parent.cond) if parent.cond is not None else ""
        return f"while ({cond}) ;"

    if isinstance(parent, c_ast.For):
        init = gen.visit(parent.init) if parent.init is not None else ""
        cond = gen.visit(parent.cond) if parent.cond is not None else ""
        nxt = gen.visit(parent.next) if parent.next is not None else ""
        return f"for ({init}; {cond}; {nxt}) ;"

    assert isinstance(parent, c_ast.DoWhile)
    cond = gen.visit(parent.cond) if parent.cond is not None else ""
    if parent.cond is not None and _node_line(parent.cond) == body_line:
        # The whole "do ... while (...);" is crammed onto one line.
        return f"do ; while ({cond});"
    return "do ;"


def _unwrap_else_preview(
    gen: c_generator.CGenerator, ast: c_ast.FileAST, if_path: Tuple[str, ...]
) -> str:
    """Build the ``+`` preview for collapsing an ``If`` down to just its
    ``else`` clause -- literally that clause's own text, unindented onto
    one line, since that's exactly what ends up where the ``if`` used to
    be (see :func:`_unwrap_else`)."""
    if_node = _get_by_path(ast, if_path)
    assert isinstance(if_node, c_ast.If), f"unwrap-else path {if_path!r} is not an If"
    return _collapse_to_one_line(gen._generate_stmt(if_node.iffalse))


def _compute_mutated_text(
    gen: c_generator.CGenerator,
    ast: c_ast.FileAST,
    all_paths: List[Tuple[str, ...]],
    line_of: Dict[Tuple[str, ...], Optional[int]],
    target_path: Tuple[str, ...],
) -> Optional[str]:
    """Compute the ``+`` replacement line for removing ``target_path``, or
    ``None`` if removing it deletes its entire physical source line (by far
    the common case -- most statements are alone on their own line).

    A replacement is needed in two situations, both arising from more than
    one statement sharing a single physical source line:

      * ``target_path`` is one of several siblings packed onto one line --
        either several statements (``ix.tab = trtab; ix.idxchain = 0; ...``)
        or several declarators in one declaration (``int32_t i, e, k;``).
        The surviving siblings are re-joined onto one line.
      * ``target_path`` is nested inside a single-line compound statement
        (the ``rd->nres = 0;`` inside ``if (i > e) { rd->nres = 0; return;
        }``) -- the enclosing statement is regenerated with just that
        nested statement removed, then collapsed back onto one line.
    """
    if target_path[-1] == _UNWRAP_ELSE_MARKER:
        return _unwrap_else_preview(gen, ast, target_path[:-1])

    if target_path[-1] in _MANDATORY_BODY_ATTRS:
        return _mandatory_body_replacement(gen, ast, target_path, line_of)

    unit_path = _find_line_unit_path(all_paths, line_of, target_path)

    if unit_path == target_path:
        # Not nested inside a larger single-line statement -- but might
        # still share its own line with siblings in the same block (a
        # multi-declarator decl, or several statements packed onto a line).
        parent_prefix = target_path[:-1]
        target_line = line_of[target_path]
        siblings = [
            p
            for p in all_paths
            if p[:-1] == parent_prefix and line_of.get(p) == target_line
        ]
        if len(siblings) <= 1:
            return None
        remaining_nodes = [_get_by_path(ast, p) for p in siblings if p != target_path]
        decl_nodes = [n for n in remaining_nodes if isinstance(n, c_ast.Decl)]
        if len(decl_nodes) == len(remaining_nodes):
            merged = _merge_decl_group(decl_nodes)
            if merged is not None:
                return merged
        return " ".join(gen._generate_stmt(n).strip() for n in remaining_nodes)

    # target_path is nested inside unit_path: regenerate just the enclosing
    # statement with the target removed, on a private copy so the shared
    # ``ast`` (still needed for every other mutant in this region) is
    # untouched.
    unit_copy = copy.deepcopy(_get_by_path(ast, unit_path))
    relative_path = target_path[len(unit_path) :]
    _remove_by_path(unit_copy, relative_path)
    return _collapse_to_one_line(gen._generate_stmt(unit_copy))


def generate_mutants(
    file_path: str,
    source: str,
    *,
    whole_file: bool = False,
    changed_lines: Optional[Set[int]] = None,
    keep_calls: Optional[Set[str]] = None,
) -> List[Mutant]:
    """Generate every statement-removal mutant for ``source``.

    ``file_path`` is used only for the ``Mutant.file`` field shown in
    reports; the actual source text to mutate is ``source``.

    By default (``whole_file=False``) this is the original marker-based
    behavior: only ``// unimut on`` / ``// unimut off`` regions are
    considered, and it is an error for none to be present.

    If ``whole_file`` is True, the entire file is treated as a single
    region (see :func:`whole_file_region`), and any marker pairs present
    are read inverted, as *exclusions* (see :func:`find_excluded_ranges`)
    rather than inclusions.

    ``changed_lines``, if given, further restricts the result to mutants
    whose line number is in that set -- this is how ``--diff`` is
    implemented: the whole file is scanned, but only mutants touching
    lines that actually changed are kept.

    ``keep_calls``, if given, is a set of function names; a statement
    that is nothing but a (possibly cast) call to one of them -- e.g.
    ``printf("%d\\n", 1 + 2);`` or ``assert(x > 0);`` -- is never offered
    as a mutant. This is how ``--keep-call`` avoids reporting that your
    logging or assertion calls "aren't tested" when what you actually
    want tested is the code around them.
    """
    mutants: List[Mutant] = []
    if whole_file:
        regions = [whole_file_region(source)]
        excluded_ranges = find_excluded_ranges(source)
    else:
        regions = find_regions(source)
        excluded_ranges = []
    source_lines = source.splitlines()

    for region in regions:
        ast, preamble_lines, _n_fake_decls, _wrapped = _parse_region(region.code)
        all_paths = _find_block_item_paths(ast)
        line_of: Dict[Tuple[str, ...], Optional[int]] = {
            p: _node_line(_get_by_path(ast, p)) for p in all_paths
        }
        gen = c_generator.CGenerator()
        for path in all_paths:
            node = _get_by_path(ast, path)
            if keep_calls and _call_name(node) in keep_calls:
                continue
            coord_line = line_of[path]
            if coord_line is None:
                continue
            region_local_line = coord_line - preamble_lines
            file_line = region.start_line + region_local_line - 1
            if whole_file and _line_excluded(file_line, excluded_ranges):
                continue
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
            mutated_display = _compute_mutated_text(gen, ast, all_paths, line_of, path)
            mutants.append(
                Mutant(
                    file=file_path,
                    line=file_line,
                    original=original_display,
                    mutated=mutated_display,
                    _apply=_make_apply(region, path),
                )
            )
    if changed_lines is not None:
        mutants = [m for m in mutants if m.line in changed_lines]
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
            // unimut on
            int a;
            int b;
            // unimut off
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
            // unimut on
            int a;
            // unimut off
            int mid;
            // unimut on
            int b;
            // unimut off
            """
        )
        regions = find_regions(src)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].code, "int a;")
        self.assertEqual(regions[1].code, "int b;")

    def test_unmatched_start_raises(self):
        with self.assertRaises(MutationError):
            find_regions("// unimut on\nint a;\n")

    def test_unmatched_stop_raises(self):
        with self.assertRaises(MutationError):
            find_regions("int a;\n// unimut off\n")

    def test_nested_start_raises(self):
        with self.assertRaises(MutationError):
            find_regions("// unimut on\n// unimut on\nint a;\n// unimut off\n")


class TestSimpleRemoval(unittest.TestCase):
    """Two independent, side-effect-free statements: removing either one
    must still produce code that compiles."""

    SRC = textwrap.dedent(
        """\
        int add(int a, int b) {
          // unimut on
          int sum = a + b;
          int doubled = sum * 2;
          // unimut off
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
          // unimut on
          int result = 0;
          if (n > 0) {
            result = 1;
          }
          // unimut off
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


class TestBareIfBody(unittest.TestCase):
    """An unbraced ``if (cond) stmt;`` body is never a ``block_items``
    entry anywhere, so without explicit handling it would never be
    offered as its own removal candidate -- only "remove the whole if"
    would show up. See ``_MANDATORY_BODY_ATTRS``."""

    SRC = textwrap.dedent(
        """\
        int classify(int n) {
          // unimut on
          int result = 0;
          if (n > 0) result = 1;
          // unimut off
          return result;
        }
        """
    )

    def test_bare_body_is_its_own_candidate(self):
        mutants = generate_mutants("classify.c", self.SRC)
        originals_and_mutated = [(m.original, m.mutated) for m in mutants]
        # "remove the whole if" (existing behavior) ...
        self.assertIn(("if (n > 0) result = 1;", None), originals_and_mutated)
        # ... and "remove just the body, still evaluate the condition"
        # (this is the fix): same source line, different mutant.
        self.assertIn(("if (n > 0) result = 1;", "if (n > 0) ;"), originals_and_mutated)

    def test_removing_bare_body_only_compiles_and_keeps_condition(self):
        mutants = generate_mutants("classify.c", self.SRC)
        mutant = next(m for m in mutants if m.mutated == "if (n > 0) ;")
        mutated_source = mutant.apply(self.SRC)
        self.assertIn("if (n > 0)", mutated_source)
        self.assertNotIn("result = 1;", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))

    def test_bare_body_on_its_own_line_shows_plain_empty_statement(self):
        # When the body is on a *different* physical line than the "if"
        # itself, the "+" preview must be scoped to just that line -- not
        # the whole (possibly multi-line) if/else reconstructed.
        src = textwrap.dedent(
            """\
            int classify(int n) {
              // unimut on
              int result = 0;
              if (n > 0)
                result = 1;
              // unimut off
              return result;
            }
            """
        )
        mutants = generate_mutants("classify.c", src)
        mutant = next(m for m in mutants if m.original == "result = 1;")
        self.assertEqual(mutant.mutated, ";")
        mutated_source = mutant.apply(src)
        self.assertIn("if (n > 0)", mutated_source)
        self.assertNotIn("result = 1;", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))


class TestElseClauseRemoval(unittest.TestCase):
    """Removing an entire ``else`` clause -- braced or not -- is a
    distinct, worthwhile mutant that (like a bare if body) never shows up
    as a ``block_items`` entry anywhere. See ``_OPTIONAL_BODY_ATTRS``."""

    BRACED_SRC = textwrap.dedent(
        """\
        int classify(int n) {
          // unimut on
          int result;
          if (n > 0) {
            result = 1;
          } else {
            result = -1;
          }
          // unimut off
          return result;
        }
        """
    )

    BARE_SRC = textwrap.dedent(
        """\
        int classify(int n) {
          // unimut on
          int result;
          if (n > 0) result = 1;
          else result = -1;
          // unimut off
          return result;
        }
        """
    )

    def test_braced_else_is_a_candidate(self):
        mutants = generate_mutants("classify.c", self.BRACED_SRC)
        originals = [m.original for m in mutants]
        self.assertIn("} else {", originals)

    def test_removing_braced_else_drops_only_the_else_branch(self):
        mutants = generate_mutants("classify.c", self.BRACED_SRC)
        mutant = next(m for m in mutants if m.original == "} else {")
        mutated_source = mutant.apply(self.BRACED_SRC)
        self.assertIn("result = 1;", mutated_source)
        self.assertNotIn("result = -1;", mutated_source)
        self.assertNotIn("else", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))

    def test_removing_whole_if_drops_both_branches_not_just_condition(self):
        # This is the *other* candidate on the same line as the "else"
        # tests above: not "remove just the else clause" (tested above),
        # but "remove the whole if" -- i.e. the entire `If` node,
        # `iftrue` and `iffalse` together, deleted from its enclosing
        # block. It must NOT turn into "the else branch runs
        # unconditionally" -- both branches disappear along with the
        # condition, leaving neither assignment behind.
        mutants = generate_mutants("classify.c", self.BRACED_SRC)
        mutant = next(m for m in mutants if m.original == "if (n > 0) {")
        self.assertIsNone(mutant.mutated)
        mutated_source = mutant.apply(self.BRACED_SRC)
        self.assertNotIn("result = 1;", mutated_source)
        self.assertNotIn("result = -1;", mutated_source)
        self.assertNotIn("else", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))

    def test_bare_else_is_a_candidate_and_removable(self):
        mutants = generate_mutants("classify.c", self.BARE_SRC)
        mutant = next(m for m in mutants if m.original == "else result = -1;")
        mutated_source = mutant.apply(self.BARE_SRC)
        self.assertIn("result = 1;", mutated_source)
        self.assertNotIn("result = -1;", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))


class TestBareLoopBody(unittest.TestCase):
    """The unbraced body of a ``while``/``for`` loop gets the same
    treatment as an unbraced ``if`` body."""

    SRC = textwrap.dedent(
        """\
        int sum_to(int n) {
          // unimut on
          int total = 0;
          int k = 0;
          while (k < n) total += k++;
          // unimut off
          return total;
        }
        """
    )

    def test_bare_while_body_is_its_own_candidate(self):
        mutants = generate_mutants("sum_to.c", self.SRC)
        originals_and_mutated = [(m.original, m.mutated) for m in mutants]
        self.assertIn(("while (k < n) total += k++;", None), originals_and_mutated)
        self.assertIn(
            ("while (k < n) total += k++;", "while (k < n) ;"),
            originals_and_mutated,
        )

    def test_removing_bare_while_body_compiles(self):
        mutants = generate_mutants("sum_to.c", self.SRC)
        mutant = next(m for m in mutants if m.mutated == "while (k < n) ;")
        mutated_source = mutant.apply(self.SRC)
        self.assertIn("while (k < n)", mutated_source)
        self.assertNotIn("total += k++", mutated_source)
        if _CC:
            self.assertTrue(_compiles(mutated_source))


class TestUnwrapElse(unittest.TestCase):
    """Collapsing an ``if``/``else`` down to just the ``else`` branch,
    executed unconditionally, dropping the condition and the ``iftrue``
    branch entirely. Distinct from removing the whole ``if`` (which drops
    the else branch too) and from removing just the else clause (which
    keeps the ``if`` guarding an empty-ish then branch)."""

    COMPOUND_ELSE_SRC = textwrap.dedent(
        """\
        void demo(void) {
          // unimut on
          if (true) {} else { foo(); bar(); }
          // unimut off
        }
        """
    )

    ELSE_IF_SRC = textwrap.dedent(
        """\
        void demo(void) {
          // unimut on
          if (true) {} else if (true) { foo(); bar(); }
          // unimut off
        }
        """
    )

    _COMPILE_PREAMBLE = (
        "typedef int bool_;\n#define true 1\nvoid foo(void);\nvoid bar(void);\n"
    )

    def test_compound_else_unwraps_to_a_bare_block(self):
        mutants = generate_mutants("demo.c", self.COMPOUND_ELSE_SRC)
        mutant = next(m for m in mutants if m.mutated == "{ foo(); bar(); }")
        mutated_source = mutant.apply(self.COMPOUND_ELSE_SRC)
        self.assertNotIn("if (true)", mutated_source)
        self.assertNotIn("else", mutated_source)
        self.assertIn("foo();", mutated_source)
        self.assertIn("bar();", mutated_source)
        if _CC:
            self.assertTrue(_compiles(self._COMPILE_PREAMBLE + mutated_source))

    def test_else_if_unwraps_to_the_inner_if(self):
        mutants = generate_mutants("demo.c", self.ELSE_IF_SRC)
        mutant = next(m for m in mutants if m.mutated == "if (true) { foo(); bar(); }")
        mutated_source = mutant.apply(self.ELSE_IF_SRC)
        self.assertNotIn("else", mutated_source)
        self.assertIn("foo();", mutated_source)
        self.assertIn("bar();", mutated_source)
        # Only one "if" should remain -- the outer one (with the always-{}
        # then-branch) is gone, collapsed into what used to be its else.
        self.assertEqual(mutated_source.count("if ("), 1)
        if _CC:
            self.assertTrue(_compiles(self._COMPILE_PREAMBLE + mutated_source))


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
          // unimut on
          TRef tra = 1;  /* first ref */
          TRef trb = 2;  // second ref
          if (tra) {
            trb = tra;
          }
          // unimut off
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


class TestWholeFileMode(unittest.TestCase):
    SRC = textwrap.dedent(
        """\
        int add(int a, int b) {
          int sum = a + b;
          int doubled = sum * 2;
          return sum;
        }

        int mul(int a, int b) {
          int product = a * b;
          return product;
        }
        """
    )

    def test_no_markers_mutates_whole_file(self):
        mutants = generate_mutants("both.c", self.SRC, whole_file=True)
        originals = [m.original for m in mutants]
        self.assertIn("int sum = a + b;", originals)
        self.assertIn("int doubled = sum * 2;", originals)
        self.assertIn("int product = a * b;", originals)

    def test_marker_based_call_still_requires_markers(self):
        # Old-style call, no whole_file: no markers present means no
        # regions at all, so no mutants -- unchanged legacy behavior.
        self.assertEqual(generate_mutants("both.c", self.SRC), [])

    def test_changed_lines_filters_to_diff(self):
        # Only line 3 ("int doubled = ...") is "changed".
        mutants = generate_mutants(
            "both.c", self.SRC, whole_file=True, changed_lines={3}
        )
        self.assertEqual(len(mutants), 1)
        self.assertEqual(mutants[0].original, "int doubled = sum * 2;")


class TestExcludedRanges(unittest.TestCase):
    SRC = textwrap.dedent(
        """\
        int add(int a, int b) {
          int sum = a + b;
          // unimut off
          int doubled = sum * 2;
          // unimut on
          return sum;
        }
        """
    )

    def test_find_excluded_ranges(self):
        ranges = find_excluded_ranges(self.SRC)
        self.assertEqual(ranges, [(3, 5)])

    def test_excluded_line_is_not_mutated(self):
        mutants = generate_mutants("add.c", self.SRC, whole_file=True)
        originals = [m.original for m in mutants]
        self.assertIn("int sum = a + b;", originals)
        self.assertNotIn("int doubled = sum * 2;", originals)

    def test_unmatched_stop_raises(self):
        src = "int x;\n// unimut off\nint y;\n"
        with self.assertRaises(MutationError):
            find_excluded_ranges(src)

    def test_start_without_stop_raises(self):
        src = "int x;\n// unimut on\nint y;\n"
        with self.assertRaises(MutationError):
            find_excluded_ranges(src)

    def test_double_stop_raises(self):
        src = "// unimut off\nint x;\n// unimut off\nint y;\n// unimut on\n"
        with self.assertRaises(MutationError):
            find_excluded_ranges(src)


class TestKeepCalls(unittest.TestCase):
    SRC = textwrap.dedent(
        """\
        void demo(int x) {
          // unimut on
          printf("%d\\n", 1 + 2);
          assert(x > 0);
          int y = x + 1;
          (void)printf("y=%d\\n", y);
          // unimut off
        }
        """
    )

    def test_no_keep_calls_includes_everything(self):
        mutants = generate_mutants("demo.c", self.SRC)
        originals = [m.original for m in mutants]
        self.assertIn('printf("%d\\n", 1 + 2);', originals)
        self.assertIn("assert(x > 0);", originals)

    def test_keep_calls_excludes_matching_statements(self):
        mutants = generate_mutants("demo.c", self.SRC, keep_calls={"printf"})
        originals = [m.original for m in mutants]
        self.assertNotIn('printf("%d\\n", 1 + 2);', originals)
        self.assertNotIn('(void)printf("y=%d\\n", y);', originals)
        # Untouched: not a printf call.
        self.assertIn("assert(x > 0);", originals)
        self.assertIn("int y = x + 1;", originals)

    def test_multiple_keep_calls(self):
        mutants = generate_mutants("demo.c", self.SRC, keep_calls={"printf", "assert"})
        originals = [m.original for m in mutants]
        self.assertNotIn('printf("%d\\n", 1 + 2);', originals)
        self.assertNotIn("assert(x > 0);", originals)
        self.assertIn("int y = x + 1;", originals)

    def test_unrelated_call_name_has_no_effect(self):
        mutants_all = generate_mutants("demo.c", self.SRC)
        mutants_filtered = generate_mutants("demo.c", self.SRC, keep_calls={"memcpy"})
        self.assertEqual(len(mutants_all), len(mutants_filtered))


class TestMutantsArePicklable(unittest.TestCase):
    """--jobs ships mutants to worker processes via pickle; make sure
    that keeps working (it broke once, when apply() was a closure)."""

    def test_mutant_survives_pickle_roundtrip(self):
        import pickle

        src = textwrap.dedent(
            """\
            int add(int a, int b) {
              // unimut on
              int sum = a + b;
              // unimut off
              return sum;
            }
            """
        )
        mutants = generate_mutants("add.c", src)
        self.assertEqual(len(mutants), 1)
        roundtripped = pickle.loads(pickle.dumps(mutants[0]))
        self.assertEqual(roundtripped.apply(src), mutants[0].apply(src))
        self.assertNotIn("int sum", roundtripped.apply(src))


class TestUnparsableRegionRaises(unittest.TestCase):
    def test_garbage_region_raises_mutation_error(self):
        src = "// unimut on\nthis is not ) ( valid c at all {{{\n// unimut off\n"
        with self.assertRaises(MutationError):
            generate_mutants("bad.c", src)


if __name__ == "__main__":
    unittest.main()
