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

## Tests

Stdlib-only (`unittest`), no dependencies:

```console
$ python3 test_torch_hostdev_lint.py -v
...
Ran 43 tests in 0.01s
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
