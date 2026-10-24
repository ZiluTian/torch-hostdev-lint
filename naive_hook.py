#!/usr/bin/env python3
"""
naive_hook.py -- the "roll your own" alternative to torch_hostdev_lint.py.

If you did NOT want to depend on this repo, the minimal pre-commit custom
rule people actually write looks like this: flag a CPU<->GPU sync call
(.item/.tolist/.numpy/.cpu) that sits inside a loop. ~80 lines of stdlib AST.

What it catches (and nothing else):
  * HIGH only: .item()/.tolist()/.numpy()/.cpu() inside for/while/comprehension
What it does NOT do (the point of the benchmark):
  * no .to(<device>) direction analysis (H2D-in-loop, .to('cpu'))
  * no torch.* factory device= placement checks
  * no severity grading, no loop context reporting, no fix suggestion
  * no suppression comments, no JSON, no CI exit-code contract

Pre-commit wiring (this is the "custom pre-commit rule" in practice -- ruff
itself has no plugin API, so custom Python checks are pre-commit hooks):

  # .pre-commit-config.yaml
  - repo: local
    hooks:
      - id: hostdev-sync-naive
        name: hostdev-sync (DIY naive rule)
        entry: python3 naive_hook.py
        language: system
        types: [python]

Exit codes: 0 = clean, 1 = finding(s) (pre-commit shows them as failures),
2 = could not read/parse a file.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SYNC_D2H = {"item", "tolist", "numpy", "cpu"}


class _LoopScanner(ast.NodeVisitor):
    def __init__(self):
        self.in_loop = 0
        self.hits = []  # (lineno, col, text)

    def _enter(self, node, kind):
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_For(self, node):
        self._enter(node, "for")

    def visit_AsyncFor(self, node):
        self._enter(node, "async-for")

    def visit_While(self, node):
        self._enter(node, "while")

    def _comp(self, node):
        self._enter(node, "comp")

    visit_ListComp = _comp
    visit_SetComp = _comp
    visit_DictComp = _comp
    visit_GeneratorExp = _comp

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in SYNC_D2H
                and self.in_loop > 0):
            self.hits.append((node.lineno, node.col_offset + 1,
                              ast.unparse(node) if hasattr(ast, "unparse") else "?"))
        self.generic_visit(node)


def findings_in_file(path: Path) -> list[tuple[int, int, str]]:
    """Return (lineno, col1based, call_text) for sync calls inside loops."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    scanner = _LoopScanner()
    scanner.visit(tree)
    scanner.hits.sort()
    return scanner.hits


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(f"{Path(__file__).name}: usage: {Path(__file__).name} FILE [FILE...]",
              file=sys.stderr)
        return 2
    any_hit = False
    errors = 0
    for arg in argv:
        p = Path(arg)
        try:
            hits = findings_in_file(p)
        except (OSError, SyntaxError) as e:
            print(f"{Path(__file__).name}: warning: {p}: {e} -- skipping",
                  file=sys.stderr)
            errors += 1
            continue
        for lineno, col, text in hits:
            print(f"{p}:{lineno}:{col}: sync-in-loop: {text}")
            any_hit = True
    if any_hit:
        return 1
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
