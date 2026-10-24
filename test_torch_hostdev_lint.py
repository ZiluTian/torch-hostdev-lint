#!/usr/bin/env python3
"""
test_torch_hostdev_lint.py -- stdlib-only tests for torch_hostdev_lint.py.

Run:  python3 test_torch_hostdev_lint.py [-v]
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_hostdev_lint as lint  # noqa: E402


def findings_for(code: str, name: str = "<test>") -> list:
    return lint.analyze_bytes(name, textwrap.dedent(code).encode("utf-8"))


def call_node(expr: str):
    return ast.parse(expr).body[0].value


class ClassifyToTests(unittest.TestCase):
    """Direction classification of .to(...): only *clear* device specs count."""

    def assert_cls(self, expr, expected):
        self.assertEqual(lint._classify_to(call_node(expr)), expected, expr)

    def test_d2h_strings(self):
        self.assert_cls("t.to('cpu')", ("transfer", "d2h"))
        self.assert_cls("t.to('cpu:0')", ("transfer", "d2h"))

    def test_h2d_strings(self):
        self.assert_cls("t.to('cuda')", ("transfer", "h2d"))
        self.assert_cls("t.to('cuda:0')", ("transfer", "h2d"))
        self.assert_cls("t.to('xpu:1')", ("transfer", "h2d"))
        self.assert_cls("t.to('mps')", ("transfer", "h2d"))

    def test_meta_is_not_a_transfer(self):
        self.assert_cls("t.to('meta')", ("n/a", "meta"))

    def test_device_name_hints(self):
        self.assert_cls("t.to(self.device)", ("transfer", "h2d"))
        self.assert_cls("t.to(dev)", ("transfer", "h2d"))
        self.assert_cls("t.to(torch.device('cpu'))", ("transfer", "d2h"))
        self.assert_cls("t.to(torch.cuda.device(0))", ("transfer", "h2d"))

    def test_ambiguous_is_maybe_not_transfer(self):
        # dtype cast, tensor cast, unknown var, bare .to() -> must NOT be H2D.
        self.assert_cls("t.to(torch.float)", ("maybe", "unknown"))
        self.assert_cls("t.to(other)", ("maybe", "unknown"))
        self.assert_cls("t.to(dtype=torch.float)", ("maybe", "unknown"))
        self.assert_cls("t.to(device='cuda', dtype=torch.float)", ("transfer", "h2d"))
        self.assertIsNone(lint._device_arg_of_to(call_node("t.to()")))


class VisitorTests(unittest.TestCase):
    def test_item_in_for_loop_is_high_with_augassign_fix(self):
        fs = findings_for(
            """
            def validate(data):
                total = 0
                for x in data:
                    total += x.item()
                return total
            """
        )
        self.assertEqual(len(fs), 1)
        f = fs[0]
        self.assertEqual(f.severity, "HIGH")
        self.assertEqual(f.kind, "sync/d2h-in-loop")
        self.assertEqual(f.lineno, 5)
        self.assertEqual(f.col, 18)  # 1-based; "        total += x.item()"
        self.assertIn("fn=validate", f.context)
        self.assertIn("for@", f.context)
        # AugAssign-specific fix: accumulate on device, sync once.
        self.assertIn(".item() ONCE after the loop", f.fix)

    def test_nested_augassign_gets_specific_fix(self):
        # `running += loss.numpy().tolist()[0]`: Call is under Subscript,
        # AugAssign is two levels up -- the walk-up must still find it.
        fs = findings_for(
            """
            def train(data):
                running = 0
                for x in data:
                    running += x.numpy().tolist()[0]
                return running
            """
        )
        # both .numpy() and .tolist() are sync calls -> two findings, and both
        # must find the enclosing AugAssign through the expression chain.
        self.assertEqual(len(fs), 2)
        for f in fs:
            self.assertIn(".item() ONCE after the loop", f.fix)

    def test_plain_statement_gets_generic_fix(self):
        fs = findings_for(
            """
            def f(data):
                for x in data:
                    print(x.item())
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertIn("Move this sync OUT of the loop", fs[0].fix)

    def test_item_outside_loop_is_low(self):
        fs = findings_for(
            """
            def report(t):
                return t.item()
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "LOW")
        self.assertEqual(fs[0].kind, "one-shot-sync/d2h")

    def test_to_cpu_in_while_loop_is_high(self):
        fs = findings_for(
            """
            def drain(buf):
                while buf:
                    x = buf.pop().to('cpu')
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "HIGH")
        self.assertEqual(fs[0].kind, "sync/d2h-in-loop")

    def test_to_device_in_loop_is_medium(self):
        fs = findings_for(
            """
            def validate(data):
                for inputs in data:
                    inputs = inputs.to(self.device)
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "MEDIUM")
        self.assertEqual(fs[0].kind, "h2d-in-loop")

    def test_comprehension_is_a_loop(self):
        fs = findings_for(
            """
            def f(a):
                return [x.item() for x in a]
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "HIGH")
        self.assertIn("list-comp@", fs[0].context)

    def test_factory_device_kwarg(self):
        in_loop = findings_for(
            """
            def step():
                for i in range(10):
                    buf = torch.zeros(3, device=dev)
            """
        )
        self.assertEqual(len(in_loop), 1)
        self.assertEqual(in_loop[0].severity, "MEDIUM")
        self.assertEqual(in_loop[0].kind, "device-placement-in-loop")

        one_shot = findings_for(
            """
            def setup():
                return torch.zeros(3, device=dev)
            """
        )
        self.assertEqual(len(one_shot), 1)
        self.assertEqual(one_shot[0].severity, "INFO")
        self.assertEqual(one_shot[0].kind, "device-placement")

    def test_from_torch_import_factory(self):
        fs = findings_for(
            """
            from torch import zeros

            def setup():
                return zeros(3, device=dev)
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "INFO")

    def test_pin_memory_is_info(self):
        fs = findings_for(
            """
            def f(x):
                return x.pin_memory()
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "INFO")
        self.assertEqual(fs[0].kind, "pin-memory")

    def test_dtype_cast_and_meta_are_not_flagged(self):
        self.assertEqual(
            findings_for(
                """
                def f(t, other):
                    return t.to(torch.float), t.to(other), t.to('meta')
                """
            ),
            [],
        )

    def test_unrelated_calls_are_not_flagged(self):
        self.assertEqual(
            findings_for(
                """
                def f(d, s):
                    return d.get('k'), s.strip(), len(d)
                """
            ),
            [],
        )

    def test_suppression_comment(self):
        fs = findings_for(
            """
            def f(a):
                for x in a:
                    y = x.item()  # hostdev-lint: ignore
            """
        )
        self.assertEqual(fs, [])

    def test_hot_path_note_on_conventional_methods(self):
        fs = findings_for(
            """
            class M:
                def forward(self, t):
                    return t.item()
            """
        )
        self.assertEqual(len(fs), 1)
        self.assertIn("conventional hot path", fs[0].context)

    def test_findings_sorted_severity_then_position(self):
        fs = findings_for(
            """
            def f(t, a):
                r = t.item()
                for x in a:
                    r += x.item()
                s = t.numpy()
                return r, s
            """
        )
        sevs = [f.severity for f in fs]
        self.assertEqual(sevs, ["HIGH", "LOW", "LOW"])

    def test_coding_cookie_and_non_ascii(self):
        data = b"# -*- coding: utf-8 -*-\n# caf\xe9 test\nx = t.item()\n"
        fs = lint.analyze_bytes("enc.py", data)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "LOW")

    def test_syntax_error_raises_with_location(self):
        with self.assertRaises(SyntaxError) as cm:
            lint.analyze_bytes("bad.py", b"def f(:\n    pass\n")
        self.assertEqual(cm.exception.lineno, 1)


class BenchmarkTemplateTests(unittest.TestCase):
    """The synthetic corpus template is ground truth for benchmark.py;
    lock its counts so template edits cannot silently skew the benchmark."""

    def test_template_ground_truth_counts(self):
        import benchmark as bench
        self.assertEqual(len(bench.TRUTH_PER_FILE), 6)
        counts = {}
        for cls, _ in bench.TRUTH_PER_FILE:
            counts[cls] = counts.get(cls, 0) + 1
        self.assertEqual(counts, bench.TRUTH_COUNTS)
        # every truth line actually carries its marker
        for cls, lineno in bench.TRUTH_PER_FILE:
            line = bench.TEMPLATE.splitlines()[lineno - 1]
            self.assertIn(f"# T:{cls}", line)

    def test_template_linter_and_hook_agree_with_truth(self):
        import benchmark as bench
        import naive_hook
        with tempfile.TemporaryDirectory() as tmp:
            files = bench.generate_corpus(Path(tmp), 2)
            linter_findings = sum(len(lint.analyze_file(f)) for f in files)
            hook_findings = sum(len(naive_hook.findings_in_file(f))
                                for f in files)
        self.assertEqual(linter_findings, 2 * sum(bench.TRUTH_COUNTS.values()))
        self.assertEqual(hook_findings, 2 * bench.TRUTH_COUNTS["HIGH"])


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        (self.root / "ok.py").write_text("def f(t):\n    for x in t:\n        r = x.item()\n")
        (self.root / "note.txt").write_text("x = t.item()\n")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "c.py").write_text("x = t.item()\n")
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "site").mkdir()
        (self.root / ".venv" / "site" / "v.py").write_text("x = t.item()\n")
        (self.root / "build").mkdir()
        (self.root / "build" / "b.py").write_text("x = t.item()\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "deep.py").write_text("def f(t):\n    for x in t:\n        r = x.item()\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ignored_dirs_are_pruned(self):
        files = lint.iter_python_files(self.root)
        self.assertEqual(
            {f.name for f in files},
            {"ok.py", "deep.py"},
        )

    def test_exclude_patterns(self):
        files = lint.iter_python_files(self.root, ("sub/*",))
        self.assertEqual({f.name for f in files}, {"ok.py"})
        # basename matching
        files = lint.iter_python_files(self.root, ("deep.py",))
        self.assertEqual({f.name for f in files}, {"ok.py"})


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_lint(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = lint.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def write(self, name, code):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(code))
        return p

    def test_clean_file_exits_zero(self):
        p = self.write("clean.py", "def f(x):\n    return x + 1\n")
        code, out, _ = self.run_lint(str(p))
        self.assertEqual(code, 0)
        self.assertIn("No host<->device", out)

    def test_high_finding_exits_one(self):
        p = self.write("hot.py",
                       "def f(t):\n    for x in t:\n        r = x.item()\n")
        code, out, _ = self.run_lint(str(p))
        self.assertEqual(code, 1)
        self.assertIn("[HIGH]", out)

    def test_low_only_finding_does_not_fail_by_default(self):
        p = self.write("low.py", "def f(t):\n    return t.item()\n")
        code, out, _ = self.run_lint(str(p))
        self.assertEqual(code, 0)  # default --fail-on HIGH
        self.assertIn("[LOW]", out)
        code, _, _ = self.run_lint(str(p), "--fail-on", "LOW")
        self.assertEqual(code, 1)

    def test_min_severity_filters_display_but_not_exit(self):
        p = self.write("mix.py",
                       "def f(t):\n    for x in t:\n        r = x.item()\n    return t.numpy()\n")
        code, out, _ = self.run_lint(str(p), "--min-severity", "HIGH")
        self.assertEqual(code, 1)  # full set still gates the exit
        self.assertIn("[HIGH]", out)
        self.assertNotIn("[LOW]", out)

    def test_quiet_summary(self):
        p = self.write("mix.py",
                       "def f(t):\n    for x in t:\n        r = x.item()\n")
        code, out, _ = self.run_lint(str(p), "--quiet")
        self.assertEqual(code, 1)
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertIn("1 finding(s) 1 HIGH", out)

    def test_json_shape(self):
        p = self.write("mix.py",
                       "def f(t):\n    for x in t:\n        r = x.item()\n")
        code, out, _ = self.run_lint(str(p), "--json")
        data = json.loads(out)
        self.assertEqual(len(data), 1)
        f = data[0]
        for key in ("file", "lineno", "col", "severity", "kind", "call",
                    "context", "message", "fix"):
            self.assertIn(key, f)
        self.assertGreaterEqual(f["col"], 1)
        self.assertGreaterEqual(f["lineno"], 1)
        self.assertEqual(f["severity"], "HIGH")
        self.assertEqual(code, 1)

    def test_json_empty_is_valid_json(self):
        p = self.write("clean.py", "x = 1\n")
        code, out, _ = self.run_lint(str(p), "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [])

    def test_nonexistent_path_is_input_error(self):
        code, _, err = self.run_lint(str(self.root / "nope.py"))
        self.assertEqual(code, 2)
        self.assertIn("no such file", err)

    def test_explicit_non_py_file_is_input_error(self):
        p = self.write("note.txt", "x = t.item()\n")
        code, _, err = self.run_lint(str(p))
        self.assertEqual(code, 2)
        self.assertIn("not a .py file", err)

    def test_empty_directory_is_input_error(self):
        code, _, err = self.run_lint(str(self.root))
        self.assertEqual(code, 2)
        self.assertIn("no Python files", err)

    def test_directory_scan_and_dedup(self):
        self.write("a.py", "def f(t):\n    for x in t:\n        r = x.item()\n")
        self.write("sub/b.py", "def f(t):\n    for x in t:\n        r = x.item()\n")
        self.write("__pycache__/c.py", "x = t.item()\n")
        # root + explicit file overlap must not double-report
        code, out, _ = self.run_lint(str(self.root), str(self.root / "a.py"), "--json")
        data = json.loads(out)
        self.assertEqual(len(data), 2)
        self.assertEqual(code, 1)

    def test_syntax_error_file_warns_and_skips(self):
        self.write("bad.py", "def f(:\n    pass\n")
        self.write("good.py", "def f(t):\n    for x in t:\n        r = x.item()\n")
        code, _, err = self.run_lint(str(self.root))
        self.assertEqual(code, 1)  # good.py still fails the run
        self.assertIn("syntax error", err)

    def test_exclude_flag(self):
        self.write("tests/test_x.py",
                   "def f(t):\n    for x in t:\n        r = x.item()\n")
        self.write("src/x.py",
                   "def f(t):\n    for x in t:\n        r = x.item()\n")
        code, out, _ = self.run_lint(str(self.root), "--exclude", "tests/*", "--json")
        data = json.loads(out)
        self.assertEqual(len(data), 1)
        self.assertIn("src", data[0]["file"])

    def test_suppressed_finding_does_not_fail(self):
        p = self.write("sup.py",
                       "def f(t):\n    for x in t:\n        r = x.item()  # hostdev-lint: ignore\n")
        code, out, _ = self.run_lint(str(p), "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [])

    def test_version_flag(self):
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                lint.main(["--version"])
        self.assertEqual(cm.exception.code, 0)


class _FakeTty(io.StringIO):
    """StringIO that claims to be a TTY, so main() thinks color is allowed."""

    def isatty(self):
        return True


class NoColorTests(unittest.TestCase):
    HOT = "def f(t):\n    for x in t:\n        r = x.item()\n"

    def _run(self, argv, extra_env=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hot.py").write_text(self.HOT)
            argv = [str(root / "hot.py")] + argv
            saved = {k: os.environ.get(k) for k in ("NO_COLOR",)}
            if extra_env:
                os.environ.update(extra_env)
            buf = _FakeTty()
            try:
                with contextlib.redirect_stdout(buf):
                    code = lint.main(argv)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            return code, buf.getvalue()

    def test_ansi_when_tty(self):
        code, out = self._run([])
        self.assertEqual(code, 1)
        self.assertIn("\033[", out)

    def test_no_color_flag_disables_ansi(self):
        code, out = self._run(["--no-color"])
        self.assertEqual(code, 1)
        self.assertNotIn("\033[", out)

    def test_no_color_env_disables_ansi(self):
        code, out = self._run([], extra_env={"NO_COLOR": "1"})
        self.assertEqual(code, 1)
        self.assertNotIn("\033[", out)

    def test_c_helper(self):
        self.assertEqual(lint._c(31, False, "x"), "x")
        self.assertEqual(lint._c(31, True, "x"), "\033[31mx\033[0m")


if __name__ == "__main__":
    unittest.main()
