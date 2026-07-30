"""Tests for ``--exit-on-first-survivor``.

Drives the whole CLI via ``unimut.main()`` (the same way
``test_recff_unpack.py`` does) rather than calling ``_run_mutants``
directly, since the behavior under test spans argument parsing, the
worker pool's early-cancellation logic, and the informational message
``main()`` prints when a run was cut short.

``--run true`` is used throughout: an always-succeeding command means
*every* mutant "survives", which is what makes the ordering
deterministic enough to test at all -- with ``--jobs 1``, mutants are
necessarily tried in the order ``generate_mutants`` produced them (the
single worker just pulls tasks off a queue in that order), so the first
one tried is always the one at index 0, and it's guaranteed to survive.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import textwrap
import unittest

from . import unimut as cli

# Several independently-removable statements, so there's more than one
# mutant to potentially stop short of.
SRC = textwrap.dedent(
    """\
    void demo(int a, int b, int c) {
      // unimut on
      a = a + 1;
      b = b + 1;
      c = c + 1;
      // unimut off
    }
    """
)


class TestExitOnFirstSurvivor(unittest.TestCase):
    def _run_cli(self, extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "demo.c")
            with open(src_path, "w") as f:
                f.write(SRC)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                out = io.StringIO()
                err = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exit_code = cli.main(
                        ["--file", "demo.c", "--run", "true", "--jobs", "1"]
                        + extra_args
                    )
            finally:
                os.chdir(cwd)
        return exit_code, out.getvalue(), err.getvalue()

    def test_without_the_flag_runs_every_mutant(self):
        exit_code, stdout, _stderr = self._run_cli(["--include-killed-mutants"])
        self.assertEqual(exit_code, 1)
        # 3 statement-removal mutants, plus 2 rhs off-by-one mutants for
        # each assignment's own value ("a + 1", "b + 1", "c + 1" are each
        # themselves a target) -- 9 total.
        self.assertIn("Survived: 9/9", stdout)

    def test_with_the_flag_stops_after_the_first_survivor(self):
        exit_code, stdout, stderr = self._run_cli(
            ["--include-killed-mutants", "--exit-on-first-survivor"]
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("Survived: 1/1", stdout)
        # Only the first mutant's own line should ever be printed.
        self.assertIn("a = a + 1;", stdout)
        self.assertNotIn("b = b + 1;", stdout)
        self.assertNotIn("c = c + 1;", stdout)

    def test_prints_an_informational_stopping_message(self):
        _exit_code, _stdout, stderr = self._run_cli(["--exit-on-first-survivor"])
        self.assertIn("stopping early", stderr)
        self.assertIn("--exit-on-first-survivor", stderr)
        # 8 of the 9 total mutants were never run.
        self.assertIn("8 mutants not run", stderr)

    def test_no_message_without_the_flag(self):
        _exit_code, _stdout, stderr = self._run_cli([])
        self.assertNotIn("stopping early", stderr)

    def test_baseline_failure_takes_priority_over_early_exit(self):
        # If --run fails even against unmodified code, that's a
        # configuration error unimut should report as such -- not
        # something --exit-on-first-survivor should paper over or
        # confuse with a normal survivor.
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "demo.c")
            with open(src_path, "w") as f:
                f.write(SRC)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                out = io.StringIO()
                err = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exit_code = cli.main(
                        [
                            "--file",
                            "demo.c",
                            "--run",
                            "false",
                            "--jobs",
                            "1",
                            "--exit-on-first-survivor",
                        ]
                    )
            finally:
                os.chdir(cwd)
        self.assertEqual(exit_code, 1)
        self.assertIn("--run failed against the unmodified code", err.getvalue())
        self.assertNotIn("stopping early", err.getvalue())


if __name__ == "__main__":
    unittest.main()
