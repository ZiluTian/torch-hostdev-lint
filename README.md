# torch-hostdev-lint

A static, **no-execution** linter for inefficient host↔device data movement in
PyTorch source. It walks the AST of your `.py` files (never imports or runs
them) and flags the API calls that force a CPU↔GPU synchronization or move data
across the bus — with enough context (enclosing function + loop) to tell a
one-shot setup call apart from a per-iteration hot-path call, and a concrete
fix for each finding.

The host↔device transfer/sync surface in torch is a small closed set, so the
linter has no false-positive machinery to tune: if you see a finding, the call
is on the list.

## What it looks for

| Class | Calls | What it means |
|---|---|---|
| **SYNC / D2H** | `.item()` `.tolist()` `.numpy()` `.cpu()` `.to('cpu')` | force a CPU↔GPU sync and/or pull tensor data to the host |
| **H2D** | `.cuda()` `.xpu()` `.to(<device>)` | move data onto a device (often per-iteration H2D) |
| **PLACE** | `torch.tensor(..., device=...)` and friends, `.pin_memory()` | device placement / host-side enablers |

### Severity (the whole point is the loop context)

| Severity | Meaning |
|---|---|
| **HIGH** | A SYNC/D2H call **inside a loop/for-loop/comprehension**. A per-iteration `.item()`/`.cpu()`/`.numpy()`/`.tolist()` serializes the whole pass — one full CPU↔GPU sync per iteration. **The anti-pattern.** |
| **MEDIUM** | An H2D transfer inside a loop. May be legitimate (fresh data each iteration) but should use `non_blocking=True` + `pin_memory=True`, and the source should not already be on device. |
| **LOW** | The same call outside any loop (one-shot setup / final sync). Usually fine; reported for awareness. |
| **INFO** | Informational (e.g. `.pin_memory()`, a `device=` factory arg). |

## Repo layout

| file | role |
|---|---|
| `torch_hostdev_lint.py` | the linter (single file, stdlib only) |
| `test_torch_hostdev_lint.py` | unittest suite (45 tests) |
| `naive_hook.py` | the minimal "roll your own" pre-commit rule, for comparison |
| `benchmark.py` | ruff vs. DIY hook vs. this linter (speed + correctness) |
| `.pre-commit-hooks.yaml` | drop-in pre-commit hook definition |

## Quick start

```console
$ python3 torch_hostdev_lint.py path/to/model.py [more.py ...]
```

Directories are scanned recursively (VCS/build/venv dirs are pruned, symlinks
not followed). Python 3.9+ required. No third-party dependencies.

```console
$ python3 torch_hostdev_lint.py examples/tuner/01_mnist/validator.py

10 finding(s) across 1 file(s): 3 MEDIUM, 7 LOW

== examples/tuner/01_mnist/validator.py ==
  examples/tuner/01_mnist/validator.py:177:35  [MEDIUM]  h2d-in-loop
      > inputs.to(self.device)
      ctx : fn=validate | for@176 (enclosing method is a conventional hot path)
      why : device transfer inside a loop
      fix : Use non_blocking=True and set pin_memory=True on the loader; ensure the source tensor is not already on the target device.

  examples/tuner/01_mnist/validator.py:200:20  [LOW]     one-shot-sync/d2h
      > (test_loss / num_batches).item()
      ctx : fn=validate (enclosing method is a conventional hot path)
      why : .item() forces a CPU<->GPU sync / host copy (one-shot).
      fix : One-shot sync is usually fine; verify it isn't on a hot path.
```

A per-iteration accumulation like `total += loss.item()` inside a loop gets a
HIGH finding with the specific fix:

> Accumulate on device instead of syncing each iteration: keep a device tensor
> (`test_loss = torch.zeros(1, device=dev); test_loss += loss`) and call
> `.item()` ONCE after the loop.

## CLI

```
usage: torch_hostdev_lint [-h] [--min-severity {HIGH,MEDIUM,LOW,INFO}]
                          [--fail-on {HIGH,MEDIUM,LOW,INFO}] [--json] [--quiet]
                          [--no-color] [--exclude PATTERN] [--version]
                          paths [paths ...]
```

| Option | Description |
|---|---|
| `--min-severity LOW` | only *display* findings at/above this severity. |
| `--fail-on HIGH` | exit non-zero if any finding at/above this severity. Operates on the **full** finding set — `--min-severity` does not gate the exit code. |
| `--json` | emit a JSON array of findings (machine-readable). |
| `--quiet` | print only the summary line. |
| `--no-color` | disable ANSI colors (also honored via the `NO_COLOR` env var). |
| `--exclude PATTERN` | glob pattern to exclude, matched against the path relative to the scan root and the basename (`*` crosses path separators). May be repeated. |

### Exit codes (CI-friendly)

| Code | Meaning |
|---|---|
| `0` | clean, or worst finding below `--fail-on` |
| `1` | at least one finding at/above `--fail-on` |
| `2` | input error: nonexistent path, explicitly passed non-`.py` file, no `.py` files found, or Python < 3.9 |

Unreadable or unparseable files are reported on stderr and skipped; they do
not affect the exit code.

### JSON schema

One object per finding:

```json
{
  "file": "examples/tuner/01_mnist/validator.py",
  "lineno": 177,
  "col": 35,
  "severity": "MEDIUM",
  "kind": "h2d-in-loop",
  "call": "inputs.to(self.device)",
  "context": "fn=validate | for@176 (enclosing method is a conventional hot path)",
  "message": "device transfer inside a loop",
  "fix": "Use non_blocking=True and set pin_memory=True on the loader; ..."
}
```

`lineno` and `col` are 1-based.

### Suppressing a finding

Put a marker comment on the call's line:

```python
total += loss.item()   # hostdev-lint: ignore
```

## Pre-commit

This repo ships a [`.pre-commit-hooks.yaml`](.pre-commit-hooks.yaml). After a
tag is cut:

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/ZiluTian/torch-hostdev-lint
  rev: v1.1.0
  hooks:
    - id: torch-hostdev-lint   # args baked in: --fail-on HIGH
```

or without a tag, via the `repo: local` variant in the file's comment.
pre-commit passes all staged `.py` files in **one invocation** (batch mode),
so per-commit cost is a single interpreter start plus the scan — see the
benchmark below for measured numbers.

## Benchmark: vs. ruff and a DIY pre-commit hook

"Can't I just add a custom rule to ruff?" — **ruff has no plugin API**, so a
custom ruff rule in the literal sense doesn't exist. The realistic DIY
alternative is a pre-commit hook running a small Python AST script; this repo
ships exactly that minimal version as [`naive_hook.py`](naive_hook.py)
(~80 lines, catches only `.item()/.tolist()/.numpy()/.cpu()` inside loops)
so the trade-off is measurable, not asserted.

[`benchmark.py`](benchmark.py) runs a three-way comparison — **ruff**
(broad real-world ruleset `E,F,W,B,S,I,C4,UP,RET,SIM`, `--isolated`),
**naive_hook**, and **this linter** — across three cost models (in-process,
batch subprocess = what pre-commit actually does, per-file subprocess):

```console
$ python3 benchmark.py            # synthetic corpus (200 files, ground truth)
$ python3 benchmark.py --files 2000
$ python3 benchmark.py --corpus path/to/pytorch   # any real tree
$ python3 benchmark.py --json
```

By default it generates a synthetic corpus in a temp dir with a known ground
truth (one of every target pattern per file, marked `# T:<CLASS>`, plus
decoys that must not be flagged) and verifies each participant against it.

Representative run (synthetic, 200 files, Python 3.12, ruff 0.16.6):

| participant | wall | findings | notes |
|---|---:|---:|---|
| ruff (batch) | 7 ms | 0 | 0 on ground-truth lines |
| torch_hostdev_lint (in-proc) | 36 ms | 1200 | full set |
| torch_hostdev_lint (batch subprocess) | 77 ms | 1200 | pre-commit cost model |
| torch_hostdev_lint (per-file, cap 50) | 1531 ms | 300 | ~30 ms/file interpreter start-up |
| naive_hook (batch subprocess) | 39 ms | 200 | HIGH-only |

How to read it:

- **ruff is ~10× faster, and catches none of it.** No built-in rule expresses
  host↔device sync, and ruff has no plugin API — speed buys you nothing you
  can't get from a 36 ms in-process scan.
- **Per-file subprocess start-up (~30 ms/file) is the only regime where the
  Python implementation costs real money**, and pre-commit does *not* run in
  that regime (one invocation per commit, all staged files). For whole-repo
  CI scans, import the module in-process instead of spawning per file.
- **The DIY hook is faster than this linter (checks less) and covers ~17% of
  the finding space** — no `.to(device)` direction analysis, no factory
  `device=` placement, no severity grading, no fix suggestions, no suppression
  contract.

Real-world scale (measured on a 4,529-file / 58 MiB compiler-toolchain repo,
pruned of `__pycache__`/`build`/caches; Python 3.12, ruff 0.16.6, median of 3):

| participant | wall | findings |
|---|---:|---:|
| ruff (batch) | 956 ms | 95,093 diagnostics, none for this pattern |
| torch_hostdev_lint (in-proc / batch) | 6.1 / 6.3 s | 1,727 |
| naive_hook (in-proc / batch) | 4.4 / 4.7 s | 180 (HIGH-only ≈ 10%) |

Unparseable files (including legacy Python 2 sources) are skipped with a
warning in every mode, matching the documented contract.

## Tests

Stdlib-only (`unittest`), no dependencies:

```console
$ python3 test_torch_hostdev_lint.py -v
...
OK
```

## Known limitations (be honest about these)

- **No type oracle.** Device placement is runtime state, not in the type
  system, so the linter cannot tell whether a given `.to()`/transfer actually
  moves data — only that the call is present and (via loop nesting) whether it
  is on a hot path. `.to()` direction is best-effort: clear device strings
  (`"cpu"`, `"cuda:0"`, `"xpu"`, `"mps"`, `"meta"`), `torch.device(<str>)`, and
  conventional names (`self.device`, `dev`, …) are classified; ambiguous
  arguments are not flagged (to avoid false positives).
- **Sync-free control flow is not flagged.** `if tensor:` or `if t.sum() > 0:`
  silently forces a sync, as do `bool(t)`/`int(t)`/`float(t)` on a tensor —
  indistinguishable from the same call on a Python scalar without a type
  oracle.
- **No interprocedural device tracking.** A variable's device is not tracked
  across function calls; "hot path" inference uses the enclosing method's name
  (e.g. `forward`, `validate`, `calibrate`) as a supplementary hint.

## License

MIT — see [LICENSE](LICENSE).
