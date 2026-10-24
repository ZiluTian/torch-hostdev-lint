#!/usr/bin/env python3
"""
torch_hostdev_lint.py -- static analyzer for inefficient host<->device data
movement in PyTorch source.

STATIC / NO-EXECUTION. It only reads source (via the `ast` module); it never
imports or runs your code. It therefore cannot *measure* cost -- it flags the
*patterns* that cause data movement / CPU-GPU synchronization, together with
enough context (function + enclosing loop) to tell a one-shot setup call apart
from a per-iteration hot-path call.

WHAT IT LOOKS FOR
-----------------
The host<->device transfer / sync API surface in torch is a small closed set,
so we walk the AST and report calls to:

  * SYNC / D2H   .item()  .tolist()  .numpy()  .cpu()  .to('cpu')
      -- force a CPU<->GPU sync and/or pull data to the host.
  * H2D          .cuda()  .xpu()  .to(<device>)
      -- move data to a device (often per-iteration H2D).
  * PLACE        torch.tensor(..., device=...) and friends, .pin_memory()
      -- device placement / host-side enablers.

SEVERITY (the whole point is the loop context)
----------------------------------------------
  HIGH    a SYNC/D2H call inside a loop/for-loop/comprehension.
          A per-iteration .item()/.cpu()/.numpy()/.tolist() serializes the
          whole pass (one full CPU-GPU sync per iteration). THE anti-pattern.
  MEDIUM  an H2D transfer inside a loop. May be legitimate (fresh data each
          iter) but should use non_blocking=True + pin_memory=True, and the
          source should not already be on device.
  LOW     the same call outside any loop (one-shot setup / final sync).
          Usually fine; reported for awareness.
  INFO    informational (e.g. .pin_memory(), a device= factory arg).

USAGE
-----
  $ python3 torch_hostdev_lint.py path/to/model.py [more.py ...] \
        [--min-severity LOW] [--fail-on HIGH] [--json] [--quiet] [--no-color] \
        [--exclude PATTERN]...

  Paths may be files or directories (scanned recursively, symlinks not
  followed). VCS/build/venv directories are always skipped (.git,
  __pycache__, .venv, venv, env, .tox, .nox, build, dist, node_modules,
  site-packages, various caches); --exclude adds glob patterns (matched
  against the path relative to the scan root AND the basename, where `*`
  crosses path separators) and may be repeated.

  A finding can be silenced with a suppression comment on the call's line:

      total += loss.item()   # hostdev-lint: ignore

  Files that cannot be read or parsed are reported on stderr and skipped;
  they do not affect the exit code.

EXIT CODES
----------
  0   no finding at/above --fail-on, and no input errors
  1   at least one finding at/above --fail-on (operates on the full finding
      set, i.e. --min-severity does NOT gate the exit code)
  2   input error: nonexistent path, explicitly passed non-.py file, no .py
      files found, or Python < 3.9

FINDINGS (JSON SCHEMA)
----------------------
  --json emits a JSON array; one object per finding:
    file, lineno (1-based), col (1-based), severity, kind, call,
    context, message, fix

KNOWN LIMITATIONS (be honest about these)
-----------------------------------------
* Device placement is runtime state, not in the type system, so we CANNOT tell
  whether a given .to()/transfer actually moves data -- only that the call is
  present and (via loop nesting) whether it is on a hot path.
* Python control-flow on a GPU value WITHOUT an explicit call (e.g.
  `if tensor:` or `if t.sum() > 0:`) silently forces a sync; that needs a type
  oracle we don't have, so it is not flagged. Likewise `bool(t)`/`int(t)`/
  `float(t)` on a tensor (indistinguishable from the same call on a Python
  scalar without a type oracle).
* No interprocedural device tracking: we do not know a variable's device across
  function calls. We only infer "hot" from the *name* of the enclosing method.

EXAMPLE
-------
On the `main` branch of ndk-env, this flags e.g. in
examples/tuner/01_mnist/validator.py:
  HIGH   test_loss += loss.item()   -> per-iteration D2H + sync in `for`
  MEDIUM inputs.to(self.device)     -> per-iteration H2D in the same loop
  ...while correctly down-grading the `.item()` / `.to(...)` calls that sit
  *after* the loop to LOW.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fnmatch
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

PROG = "torch_hostdev_lint"
__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# Configuration: the closed set of API calls that move data / force a sync.
# --------------------------------------------------------------------------- #

# Force a CPU<->GPU sync and/or pull tensor data back to the host.
SYNC_D2H = {"item", "tolist", "numpy", "cpu"}

# Move data onto a device.
H2D = {"cuda", "xpu"}

# `.to(<device>)` is a transfer; we classify the argument to know direction.
TO = "to"

# Host-side pinned-memory enabler (not a cost by itself, but relevant).
PIN_MEMORY = "pin_memory"

# torch.* factory calls whose `device=` kwarg places data on a device.
FACTORY_DEVICE = {
    "tensor", "as_tensor", "from_numpy", "empty", "zeros", "ones", "full",
    "arange", "linspace", "rand", "randn", "randint", "eye", "meshgrid",
}

# String prefixes for device specs handed to .to(<str>).
D2H_STR_PREFIXES = ("cpu",)
H2D_STR_PREFIXES = ("cuda", "gpu", "xpu", "mps")
META_STR_PREFIXES = ("meta",)

# Names that strongly suggest a torch.device value (for `.to(<name>)`).
DEVICE_HINTS = {
    "device", "self.device", "dev", "self.dev", "gpu", "self.gpu",
    "target", "self.target", "accel", "accelerator", "self.accelerator",
    "torch.device", "torch.cuda", "torch.xpu",
    "torch.cuda.device", "torch.xpu.device",
}

# Method names that are conventionally per-iteration / hot (supplementary hint).
HOT_NAMES = {
    "forward", "__call__", "step", "train_step", "eval_step", "training_step",
    "validation_step", "validate", "calibrate", "infer", "predict", "backward",
}

SEVERITIES = ("HIGH", "MEDIUM", "LOW", "INFO")
SEV_RANK = {sev: 3 - i for i, sev in enumerate(SEVERITIES)}

# Directories that never contain user source worth scanning (pruned always).
DEFAULT_IGNORED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".tox", ".nox", ".eggs", ".venv", "venv", "env",
    "node_modules", "build", "dist", "site-packages",
}

# A finding whose line carries one of these markers is suppressed.
SUPPRESS_MARKERS = ("hostdev-lint: ignore",)


@dataclasses.dataclass
class Finding:
    file: str
    lineno: int   # 1-based
    col: int      # 1-based
    severity: str
    kind: str     # short tag
    call: str     # the call text
    context: str  # enclosing function + loop headers
    message: str
    fix: str

    def as_dict(self):
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Small AST helpers
# --------------------------------------------------------------------------- #

def _unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _build_parent_map(tree):
    """Map id(node) -> parent node, for the whole tree.

    Valid while `tree` (and thus every node in it) stays alive; the only
    shared-id case in CPython is the singleton Load/Store/Del context nodes,
    which this linter never queries.
    """
    pmap = {id(tree): None}
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            pmap[id(child)] = node
            stack.append(child)
    return pmap


def _extract_name(expr) -> Optional[str]:
    """Best-effort dotted name for an expression, e.g. `self.device`."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _extract_name(expr.value)
        return f"{base}.{expr.attr}" if base else None
    if isinstance(expr, ast.Call):  # torch.device(...) / torch.cuda.device(...)
        return _extract_name(expr.func) if hasattr(expr, "func") else None
    return None


def _device_arg_of_to(call: ast.Call):
    """Return (device_expr | None). Finds the device target of a .to(...) call."""
    for kw in call.keywords:
        if kw.arg in ("device", "dst_type"):
            return kw.value
    if call.args:
        return call.args[0]
    return None


def _classify_device_str(v: str):
    """Classify a device-spec string (from .to(<str>) or torch.device(<str>))."""
    v = v.lower()
    if v.startswith(D2H_STR_PREFIXES):
        return "transfer", "d2h"
    if v.startswith(H2D_STR_PREFIXES):
        return "transfer", "h2d"
    if v.startswith(META_STR_PREFIXES):
        return "n/a", "meta"
    return "maybe", v


def _classify_to(call: ast.Call):
    """Classify a `.to(...)` call -> ('transfer'|'maybe'|'n/a', 'h2d'|'d2h'|'unknown'|...)."""
    dev = _device_arg_of_to(call)
    if dev is None:
        return "maybe", "unknown"  # .to(dtype) or .to(tensor) or .to()
    if isinstance(dev, ast.Constant) and isinstance(dev.value, str):
        return _classify_device_str(dev.value)
    # torch.device(<str>) / device(<str>): the inner string settles the direction.
    if isinstance(dev, ast.Call):
        fname = _extract_name(dev.func)
        if fname in ("torch.device", "device") and dev.args:
            inner = dev.args[0]
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                return _classify_device_str(inner.value)
            return ("transfer", "h2d") if fname in DEVICE_HINTS else ("maybe", "unknown")
    name = _extract_name(dev)
    if name and name in DEVICE_HINTS:
        return "transfer", "h2d"
    # a tensor / unknown object as arg -> matching device/dtype (possible d2d)
    return "maybe", "unknown"


# --------------------------------------------------------------------------- #
# The visitor
# --------------------------------------------------------------------------- #

class HostDevVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines, parent_map: dict):
        self.filename = filename
        self.lines = lines
        self.parent = parent_map
        self.findings: list[Finding] = []
        self.func_stack = []
        self.loop_stack = []  # each: (kind, header, lineno)

    # -- context tracking -------------------------------------------------- #
    def _push_func(self, node):
        self.func_stack.append(node.name)

    def _pop_func(self):
        self.func_stack.pop()

    def _push_loop(self, node, kind, header):
        self.loop_stack.append((kind, header, node.lineno))

    def _pop_loop(self):
        self.loop_stack.pop()

    def _src_line(self, lineno: int) -> str:
        try:
            return self.lines[lineno - 1].strip()
        except IndexError:
            return ""

    def _is_suppressed(self, lineno: int) -> bool:
        line = self._src_line(lineno)
        return any(marker in line for marker in SUPPRESS_MARKERS)

    def _context_str(self, node) -> str:
        parts = []
        if self.func_stack:
            parts.append(f"fn={self.func_stack[-1]}")
        for kind, header, lineno in reversed(self.loop_stack):
            parts.append(f"{kind}@{lineno}")
        return " | ".join(parts) if parts else "module-level"

    def _hot_path_note(self) -> str:
        if self.func_stack and self.func_stack[-1] in HOT_NAMES:
            return " (enclosing method is a conventional hot path)"
        return ""

    # -- function / loop dispatch ----------------------------------------- #
    def visit_FunctionDef(self, node):
        self._push_func(node)
        self.generic_visit(node)
        self._pop_func()

    def visit_AsyncFunctionDef(self, node):
        self._push_func(node)
        self.generic_visit(node)
        self._pop_func()

    def visit_For(self, node):
        self._push_loop(node, "for", self._src_line(node.lineno))
        self.generic_visit(node)
        self._pop_loop()

    def visit_AsyncFor(self, node):
        self._push_loop(node, "async-for", self._src_line(node.lineno))
        self.generic_visit(node)
        self._pop_loop()

    def visit_While(self, node):
        self._push_loop(node, "while", self._src_line(node.lineno))
        self.generic_visit(node)
        self._pop_loop()

    # comprehensions are also loops (their bodies run per-element)
    def _visit_comprehension(self, node, kind):
        self._push_loop(node, kind, f"<{kind}>")
        self.generic_visit(node)
        self._pop_loop()

    def visit_ListComp(self, node):
        self._visit_comprehension(node, "list-comp")

    def visit_SetComp(self, node):
        self._visit_comprehension(node, "set-comp")

    def visit_DictComp(self, node):
        self._visit_comprehension(node, "dict-comp")

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node, "genexp")

    # -- the interesting node: calls -------------------------------------- #
    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute):
            base = f.value
            if (isinstance(base, ast.Name) and base.id == "torch"
                    and f.attr in FACTORY_DEVICE):
                # torch.tensor(...) & friends: device= placement, not a method.
                self._handle_factory(node, f.attr, f"torch.{f.attr}")
            else:
                self._handle_method_call(node, f.attr)
        elif isinstance(f, ast.Name) and f.id in FACTORY_DEVICE:
            # `from torch import tensor` style: plain-name factory.
            self._handle_factory(node, f.id, f.id)

        self.generic_visit(node)

    # -- classification + reporting --------------------------------------- #
    def _handle_method_call(self, node, attr):
        in_loop = len(self.loop_stack) > 0
        ctx = self._context_str(node)

        if attr in SYNC_D2H or (attr == TO and self._is_d2h_to(node)):
            sev = "HIGH" if in_loop else "LOW"
            kind = "sync/d2h-in-loop" if in_loop else "one-shot-sync/d2h"
            msg = (
                f".{attr}() forces a CPU<->GPU sync / host copy"
                f"{' EVERY loop iteration' if in_loop else ' (one-shot)'}."
            )
            fix = self._fix_for_sync(node, in_loop)
        elif attr in H2D or (attr == TO and self._is_h2d_to(node)):
            sev = "MEDIUM" if in_loop else "LOW"
            kind = "h2d-in-loop" if in_loop else "one-shot-h2d"
            msg = (
                f"device transfer inside a loop" if in_loop
                else "one-shot device transfer"
            )
            fix = (
                "Use non_blocking=True and set pin_memory=True on the loader; "
                "ensure the source tensor is not already on the target device."
            )
        elif attr == PIN_MEMORY:
            sev = "INFO"
            kind = "pin-memory"
            msg = "host-side pinned-memory allocation (enabler for async H2D)."
            fix = "Keep if the matching .to(device) uses non_blocking=True."
        else:
            return  # not in our set

        self._add(node, sev, kind, msg, fix, ctx)

    def _is_d2h_to(self, node):
        return _classify_to(node) == ("transfer", "d2h")

    def _is_h2d_to(self, node):
        cls = _classify_to(node)
        # count only *clear* device transfers as H2D; ambiguous -> skip (avoid FPs)
        return cls == ("transfer", "h2d")

    def _handle_factory(self, node, name, qualified):
        for kw in node.keywords:
            if kw.arg == "device":
                in_loop = len(self.loop_stack) > 0
                sev = "MEDIUM" if in_loop else "INFO"
                kind = "device-placement-in-loop" if in_loop else "device-placement"
                msg = f"{qualified}(..., device=...) allocates on a device" \
                      f"{' every iteration' if in_loop else ' (one-shot)'}. " \
                      f"Check that the source data is not being H2D-copied per iter."
                fix = "If the source is CPU data moved each iter, pre-place it on " \
                      "device once, or stream it with a pinned async loader."
                self._add(node, sev, kind, msg, fix, self._context_str(node))
                break

    def _fix_for_sync(self, node, in_loop):
        if not in_loop:
            return "One-shot sync is usually fine; verify it isn't on a hot path."
        # Walk up the expression chain to the enclosing statement: an augmented
        # assignment of the synced value (`total += x.item()`, possibly nested
        # like `total += x.numpy().tolist()[0]`) is the canonical
        # "accumulate on device" case.
        cur = node
        for _ in range(8):  # expression nesting is shallow; bound the walk
            parent = self.parent.get(id(cur))
            if isinstance(parent, ast.AugAssign):
                return ("Accumulate on device instead of syncing each iteration: "
                        "keep a device tensor (test_loss = torch.zeros(1, device=dev); "
                        "test_loss += loss) and call .item() ONCE after the loop.")
            if not isinstance(parent, ast.expr):
                break
            cur = parent
        return ("Move this sync OUT of the loop: buffer results in a device tensor "
                "and convert/sync once after the loop ends.")

    # -- emit ------------------------------------------------------------- #
    def _add(self, node, sev, kind, msg, fix, ctx):
        if self._is_suppressed(node.lineno):
            return
        self.findings.append(Finding(
            file=self.filename,
            lineno=node.lineno,
            col=getattr(node, "col_offset", 0) + 1,  # 1-based, as displayed
            severity=sev,
            kind=kind,
            call=_unparse(node),
            context=ctx + self._hot_path_note(),
            message=msg,
            fix=fix,
        ))


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def _excluded(rel_path: str, name: str, patterns) -> bool:
    """True if rel_path/name matches any user --exclude glob (fnmatch-style)."""
    return any(fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(name, pat)
               for pat in patterns)


def iter_python_files(root: Path, exclude_patterns=()) -> list[Path]:
    """All .py files under root (recursive, deterministic order).

    Returned paths keep the caller's spelling of `root` (relative stays
    relative). Prunes DEFAULT_IGNORED_DIRS and user-excluded directories at
    every depth; never follows directory symlinks.
    """
    root_str = str(root)
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_str, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root_str)
        keep_dirs = []
        for d in dirnames:
            if d in DEFAULT_IGNORED_DIRS:
                continue
            d_rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if _excluded(d_rel, d, exclude_patterns):
                continue
            keep_dirs.append(d)
        dirnames[:] = sorted(keep_dirs)
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            rel = fn if rel_dir == "." else f"{rel_dir}/{fn}"
            if _excluded(rel, fn, exclude_patterns):
                continue
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Analysis entry point
# --------------------------------------------------------------------------- #

def analyze_bytes(name: str, data: bytes) -> list[Finding]:
    """Parse `data` (a Python source file named `name`) and return findings.

    Raises SyntaxError (with filename/line info) on unparseable input.
    """
    # Parse from bytes so PEP 263 coding declarations are honored; suppress
    # SyntaxWarning noise from (mostly third-party) invalid escape sequences.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(data, filename=name)
    lines = data.decode("utf-8", "replace").splitlines()
    visitor = HostDevVisitor(name, lines, _build_parent_map(tree))
    visitor.visit(tree)
    visitor.findings.sort(key=lambda f: (-SEV_RANK[f.severity], f.lineno, f.col))
    return visitor.findings


def analyze_file(path: Path) -> list[Finding]:
    """Analyze one file. Raises OSError on read failure, SyntaxError on bad syntax."""
    return analyze_bytes(str(path), path.read_bytes())


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_BAR = {
    "HIGH":   ("[HIGH]   ", 31, "red"),
    "MEDIUM": ("[MEDIUM] ", 33, "yellow"),
    "LOW":    ("[LOW]    ", 36, "cyan"),
    "INFO":   ("[INFO]   ", 90, "gray"),
}


def _c(code, use_color, text):
    return f"\033[{code}m{text}\033[0m" if use_color else text


def _sev_counts(findings) -> dict:
    counts = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts


def _summary_line(findings) -> str:
    counts = _sev_counts(findings)
    summary = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if s in counts)
    return f"{len(findings)} finding(s) {summary}" if findings else "No findings."


def print_text(findings, use_color=True):
    if not findings:
        print("No host<->device data-movement hotspots found.")
        return
    by_file = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)

    counts = _sev_counts(findings)
    summary = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if s in counts)
    print(f"\n{len(findings)} finding(s) across {len(by_file)} file(s): {summary}\n")

    for fp in sorted(by_file):
        print(f"== {fp} ==")
        for f in by_file[fp]:
            label, code, _ = _BAR[f.severity]
            print(f"  {fp}:{f.lineno}:{f.col}  {_c(code, use_color, label)} {f.kind}")
            print(f"      > {f.call}")
            print(f"      ctx : {f.context}")
            print(f"      why : {f.message}")
            print(f"      fix : {f.fix}")
            print()


def print_json(findings):
    print(json.dumps([f.as_dict() for f in findings], indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _warn(msg: str):
    print(f"{PROG}: warning: {msg}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Static (no-execution) linter for inefficient host<->device "
                    "data movement in PyTorch source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Severity: HIGH = sync/D2H inside a loop; MEDIUM = H2D inside a "
            "loop; LOW = one-shot; INFO = enablers.\n"
            "Suppress a finding with a comment on its line:  # hostdev-lint: ignore\n"
            "Exit codes: 0 clean / below --fail-on, 1 finding at/above --fail-on, "
            "2 input error (bad path, no files found, Python < 3.9).\n"
            "Known limitations: no type oracle (device of a value is runtime "
            "state), so .to() direction is best-effort and sync-free control "
            "flow (if tensor:) is not flagged."
        ))
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("--min-severity", choices=SEVERITIES, default="LOW",
                    help="only report findings at or above this severity (default LOW)")
    ap.add_argument("--fail-on", choices=SEVERITIES, default="HIGH",
                    help="exit non-zero if any finding at/above this severity (default HIGH)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                    help="glob pattern to exclude, matched against the path relative "
                         "to the scan root and the basename (may be repeated); "
                         "VCS/build/venv directories are always skipped")
    ap.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    args = ap.parse_args(argv)

    if sys.version_info < (3, 9):
        _warn(f"Python 3.9+ is required (ast.unparse); got {sys.version.split()[0]}")
        return 2

    use_color = (not args.no_color) and os.environ.get("NO_COLOR") is None \
        and sys.stdout.isatty()
    min_rank = SEV_RANK[args.min_severity]
    fail_rank = SEV_RANK[args.fail_on]

    # Collect the file list: explicit files validated, directories walked, dedup.
    files: list[Path] = []
    seen = set()
    input_errors = 0

    def add_file(p: Path):
        nonlocal input_errors
        # realpath: normalize ".." segments and resolve symlinks for dedup only;
        # the reported paths keep the user's spelling.
        key = os.path.realpath(p)
        if key in seen:
            return
        if not p.exists():
            _warn(f"{p}: no such file or directory -- skipping")
            input_errors += 1
            return
        if p.is_file() and p.suffix != ".py":
            _warn(f"{p}: not a .py file -- skipping")
            input_errors += 1
            return
        seen.add(key)
        files.append(p)

    for p in args.paths:
        pp = Path(p)
        if pp.is_dir():
            for f in iter_python_files(pp, args.exclude):
                add_file(f)
        else:
            add_file(pp)

    if not files:
        _warn("no Python files found to scan")
        return 2

    all_findings = []
    for fp in files:
        try:
            all_findings += analyze_file(fp)
        except SyntaxError as e:
            _warn(f"{fp}: syntax error ({e.msg} at line {e.lineno}) -- skipping")
        except (OSError, UnicodeDecodeError) as e:
            _warn(f"{fp}: cannot read ({e}) -- skipping")

    shown = [f for f in all_findings if SEV_RANK[f.severity] >= min_rank]

    if args.json:
        print_json(shown)
    elif args.quiet:
        print(_summary_line(shown))
    else:
        print_text(shown, use_color=use_color)

    if input_errors:
        return 2
    # exit code driven by --fail-on (operates on the FULL set, not the filtered view)
    worst = max((SEV_RANK[f.severity] for f in all_findings), default=0)
    return 1 if worst >= fail_rank else 0


if __name__ == "__main__":
    sys.exit(main())
