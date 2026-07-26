"""End-to-end regression test for unimut against the real LuaJIT
``recff_unpack`` recorder function -- the exact function unimut's C
backend was built to handle (unknown types like ``TRef``/``jit_State``,
a ``LJ_FASTCALL`` calling-convention macro, IR-emission macros used as
plain function calls, comments interleaved with statements, and nested
blocks).

Unlike ``mutate_c``'s own unit tests (which call ``generate_mutants()``
directly), this drives the whole CLI via ``unimut.main()``: argument
parsing, mutant generation, the worker pool, and ``_print_report`` --
with ``--include-killed-mutants`` so the report lists every mutant
unimut generates for this function, not just survivors. ``--run true``
is a trivial always-succeeding command; it doesn't try to validate any
real LuaJIT semantics, it exists purely so every generated mutant
"survives" and therefore is guaranteed to show up in the report. That
turns the report into a plain enumeration of the 35 statement-removal
candidates unimut finds in this function -- the exact reference count
from the README's own ``recff_unpack`` example (``Survived: 2/35``,
there run against a real test suite) -- which is what gets diffed
against ``EXPECTED_REPORT`` below.

If a future change to ``mutate_c``'s parsing heuristics or statement
walk ever adds, drops, or reorders a candidate for this real-world
function, this test fails with a unified diff pinpointing exactly which
line changed.

Run with:
    python -m unittest unimut.test_recff_unpack -v
"""

from __future__ import annotations

import contextlib
import difflib
import io
import os
import tempfile
import textwrap
import unittest

from . import unimut as cli

# The real recff_unpack body, verbatim, wrapped in unimut markers exactly
# as the README's own example does (markers around the whole function,
# signature included).
RECFF_UNPACK_SRC = textwrap.dedent(
    """\
    // unimut on
    /* unpack(t, [i, [j]]) */
    static void LJ_FASTCALL recff_unpack(jit_State *J, RecordFFData *rd)
    {
      TRef trtab = J->base[0];
      TRef tri = J->base[1];
      TRef trj = J->base[2];
      RecordIndex ix;
      GCtab *t;
      int32_t i, e, k;
      if (!tref_istab(trtab)) return;  /* Interpreter will throw. */
      t = tabV(&rd->argv[0]);
      if (tref_isnil(tri)) i = 1;
      else {
        i = argv2int(J, &rd->argv[1]);
        if (tref_isk(tri))
          emitir(IRTGI(IR_EQ), tri, lj_ir_kint(J, i));
      }
      if (!tref_isnil(trj)) {  /* trj set guarantees tri was too. */
        e = argv2int(J, &rd->argv[2]);
        if (!tref_isk(trj))
          emitir(IRTGI(IR_EQ), trj, lj_ir_kint(J, e));
      } else {  /* Guard the length, since it wasn't given as a constant. */
        TRef trlen = emitir(IRTI(IR_ALEN), trtab, TREF_NIL);
        e = (int32_t)lj_tab_len(t);
        emitir(IRTGI(IR_EQ), trlen, lj_ir_kint(J, e));
      }
      if (i > e) { rd->nres = 0; return; }
      int32_t maxn = LJ_MAX_JSLOTS - (int32_t)J->baseslot;
      uint32_t span = (uint32_t)e - (uint32_t)i;  /* n - 1, exact and overflow-free even at INT32_MIN/MAX. */
      if (maxn <= 0 || span >= (uint32_t)maxn)
        lj_trace_err_info(J, LJ_TRERR_STACKOV);
      int32_t n = (int32_t)span + 1;  /* safe: span < maxn <= LJ_MAX_JSLOTS here. */
      ix.tab = trtab; ix.idxchain = 0; ix.val = 0;
      settabV(J->L, &ix.tabv, t);
      rd->nres = n;
      for (k = 0; k < n; k++) {
        ix.key = lj_ir_kint(J, i + k);
        setintV(&ix.keyv, i + k);
        J->base[k] = lj_record_idx(J, &ix);
      }
    }
    // unimut off
"""
)

# Captured once from an actual `unimut --file lj_ffrecord.c --run true
# --include-killed-mutants` run against RECFF_UNPACK_SRC above, then
# frozen here as the reference. mutate_c auto-synthesizes fake typedefs
# for TRef/jit_State/RecordFFData/RecordIndex/GCtab (all unknown to
# pycparser) and strips the LJ_FASTCALL macro on its own -- no manual
# preamble was needed to get this function to parse.
EXPECTED_REPORT = textwrap.dedent(
    """\
    lj_ffrecord.c:5
    - TRef trtab = J->base[0];

    lj_ffrecord.c:6
    - TRef tri = J->base[1];

    lj_ffrecord.c:7
    - TRef trj = J->base[2];

    lj_ffrecord.c:8
    - RecordIndex ix;

    lj_ffrecord.c:9
    - GCtab *t;

    lj_ffrecord.c:10
    - int32_t i, e, k;

    lj_ffrecord.c:10
    - int32_t i, e, k;

    lj_ffrecord.c:10
    - int32_t i, e, k;

    lj_ffrecord.c:11
    - if (!tref_istab(trtab)) return;  /* Interpreter will throw. */

    lj_ffrecord.c:12
    - t = tabV(&rd->argv[0]);

    lj_ffrecord.c:13
    - if (tref_isnil(tri)) i = 1;

    lj_ffrecord.c:15
    - i = argv2int(J, &rd->argv[1]);

    lj_ffrecord.c:16
    - if (tref_isk(tri))

    lj_ffrecord.c:19
    - if (!tref_isnil(trj)) {  /* trj set guarantees tri was too. */

    lj_ffrecord.c:20
    - e = argv2int(J, &rd->argv[2]);

    lj_ffrecord.c:21
    - if (!tref_isk(trj))

    lj_ffrecord.c:24
    - TRef trlen = emitir(IRTI(IR_ALEN), trtab, TREF_NIL);

    lj_ffrecord.c:25
    - e = (int32_t)lj_tab_len(t);

    lj_ffrecord.c:26
    - emitir(IRTGI(IR_EQ), trlen, lj_ir_kint(J, e));

    lj_ffrecord.c:28
    - if (i > e) { rd->nres = 0; return; }

    lj_ffrecord.c:28
    - if (i > e) { rd->nres = 0; return; }

    lj_ffrecord.c:28
    - if (i > e) { rd->nres = 0; return; }

    lj_ffrecord.c:29
    - int32_t maxn = LJ_MAX_JSLOTS - (int32_t)J->baseslot;

    lj_ffrecord.c:30
    - uint32_t span = (uint32_t)e - (uint32_t)i;  /* n - 1, exact and overflow-free even at INT32_MIN/MAX. */

    lj_ffrecord.c:31
    - if (maxn <= 0 || span >= (uint32_t)maxn)

    lj_ffrecord.c:33
    - int32_t n = (int32_t)span + 1;  /* safe: span < maxn <= LJ_MAX_JSLOTS here. */

    lj_ffrecord.c:34
    - ix.tab = trtab; ix.idxchain = 0; ix.val = 0;

    lj_ffrecord.c:34
    - ix.tab = trtab; ix.idxchain = 0; ix.val = 0;

    lj_ffrecord.c:34
    - ix.tab = trtab; ix.idxchain = 0; ix.val = 0;

    lj_ffrecord.c:35
    - settabV(J->L, &ix.tabv, t);

    lj_ffrecord.c:36
    - rd->nres = n;

    lj_ffrecord.c:37
    - for (k = 0; k < n; k++) {

    lj_ffrecord.c:38
    - ix.key = lj_ir_kint(J, i + k);

    lj_ffrecord.c:39
    - setintV(&ix.keyv, i + k);

    lj_ffrecord.c:40
    - J->base[k] = lj_record_idx(J, &ix);

    Survived: 35/35
    """
)


class TestRecffUnpackMutantList(unittest.TestCase):
    """Runs the real CLI against the real function and diffs the report."""

    def _run_cli_report(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "lj_ffrecord.c")
            with open(src_path, "w") as f:
                f.write(RECFF_UNPACK_SRC)

            # Not a git checkout, so unimut's repo-root detection falls
            # back to the current directory -- chdir here first so that
            # fallback (and therefore the per-worker repo copy) stays
            # scoped to this one-file temp dir instead of picking up
            # whatever the real working directory happens to contain.
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.exit_code = cli.main(
                        [
                            "--file",
                            "lj_ffrecord.c",
                            "--run",
                            "true",
                            "--include-killed-mutants",
                        ]
                    )
            finally:
                os.chdir(cwd)
        return captured.getvalue()

    def test_full_mutant_list_matches_expected(self):
        actual = self._run_cli_report()
        if actual != EXPECTED_REPORT:
            diff = "".join(
                difflib.unified_diff(
                    EXPECTED_REPORT.splitlines(keepends=True),
                    actual.splitlines(keepends=True),
                    fromfile="expected",
                    tofile="actual",
                )
            )
            self.fail(f"mutant report for recff_unpack differs:\n{diff}")

        # --run true never fails, so nothing gets killed: every one of
        # the 35 mutants "survives", which is also why the exit code is
        # 1 here (unimut's CLI convention: nonzero iff something
        # survived) rather than a sign anything is actually wrong.
        self.assertEqual(self.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
