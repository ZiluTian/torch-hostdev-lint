#!/usr/bin/env python3
"""
benchmark.py -- compare torch_hostdev_lint against the realistic alternatives.

The question this answers: "why not just add a custom pre-commit rule to
ruff?" The honest answer, measured:

  * ruff has NO plugin API -- a "custom ruff rule" in the literal sense does
    not exist. The realistic DIY alternative is a pre-commit hook that runs a
    small Python AST script (naive_hook.py in this repo).
  * So we benchmark three participants:
      A. ruff          -- Rust speed baseline; best-effort real-world ruleset
                          (E,F,W,B,S,I,C4,UP,RET,SIM). It cannot express this
                          check at all; we measure it anyway (speed + how much
                          of the ground truth any built-in rule happens to
                          touch).
      B. naive_hook    -- the "roll your own" pre-commit rule: ~80 lines of
                          stdlib AST, catches ONLY the canonical HIGH
                          anti-pattern (.item/.tolist/.numpy/.cpu in a loop).
      C. this linter   -- full closed set, loop-context severity, fixes, JSON,
                          CI exit codes, suppression.
  * Three cost models for B and C (the Python tools):
      in-process batch   -- import once, analyze all files (IDE plugin / CI
                            script cost)
      batch subprocess   -- one interpreter start, all files as args (this is
                            exactly what pre-commit does: one hook invocation
                            per commit with the staged files)
      per-file subprocess -- one interpreter start per file (upper bound;
                            capped by --per-file-cap)

CORPUS
------
By default a SYNTHETIC corpus is generated in a temp dir: N identical files,
each containing exactly one of every target pattern (marked `# T:<CLASS>`)
plus decoy calls that must NOT be flagged (`# D`). That gives independent
ground truth for the correctness table, and a scaling axis via --files.
Pass --corpus PATH to run on a real tree (e.g. a PyTorch checkout); the
correctness table is skipped then (no ground truth), only speed + counts.

USAGE
-----
  $ python3 benchmark.py                     # synthetic, 200 files
  $ python3 benchmark.py --files 2000        # scale the synthetic corpus
  $ python3 benchmark.py --corpus path/to/pytorch   # real tree
  $ python3 benchmark.py --json              # machine-readable results

  $ python3 benchmark.py --ruff /path/to/ruff --runs 9 --per-file-cap 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import naive_hook  # noqa: E402
import torch_hostdev_lint as lint  # noqa: E402

LINTER = HERE / "torch_hostdev_lint.py"
HOOK = HERE / "naive_hook.py"

# Realistic pre-commit ruleset for ruff (the broadest a sane config uses).
RUFF_SELECT = "E,F,W,B,S,I,C4,UP,RET,SIM"
ARGV_CHUNK = 1000  # max files per subprocess argv


# --------------------------------------------------------------------------- #
# Synthetic corpus (with ground truth)
# --------------------------------------------------------------------------- #

TEMPLATE = """\
import torch


def setup(t, other, dev):
    y = t.item()                # T:LOW
    z = t.to(dev)               # T:LOW
    p = t.pin_memory()          # T:INFO
    c1 = t.to(torch.float)      # D
    c2 = t.to(other)            # D
    return y, z, p, c1, c2


def step(data, dev):
    total = 1
    acc = None
    buf = None
    for x in data:
        total += x.item()       # T:HIGH
        acc = x.to(dev)         # T:MEDIUM
    while total > 0:
        buf = torch.zeros(3, device=dev)  # T:MEDIUM
        total -= 1
    return total, acc, buf
"""

# Ground truth: (class, line) pairs, computed from the markers (independent of
# any linter logic).
TRUTH_PER_FILE: list[tuple[str, int]] = []
for _lineno, _line in enumerate(TEMPLATE.splitlines(), start=1):
    m = _line.find(" # T:")
    if m != -1:
        _cls = _line[m + 5:m + 11].strip()
        assert _cls in ("HIGH", "MEDIUM", "LOW", "INFO"), (_lineno, _cls)
        TRUTH_PER_FILE.append((_cls, _lineno))
TRUTH_PER_FILE.sort(key=lambda t: t[1])
# canonical order for reporting
TRUTH_CLASSES = ["HIGH", "MEDIUM", "LOW", "INFO"]
TRUTH_COUNTS = {c: sum(1 for cl, _ in TRUTH_PER_FILE if cl == c)
                for c in TRUTH_CLASSES}
assert sum(TRUTH_COUNTS.values()) == len(TRUTH_PER_FILE) == 6, TRUTH_PER_FILE


def generate_corpus(root: Path, n_files: int) -> list[Path]:
    files = []
    for i in range(n_files):
        p = root / f"mod_{i:05d}.py"
        p.write_text(TEMPLATE)
        files.append(p)
    return files


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

def time_median(fn, runs: int, warmup: int = 1):
    """Return (median_secs, all_samples, last_result)."""
    last = None
    for _ in range(warmup):
        last = fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples, last


def run_cmd(cmd: list[str]) -> tuple[float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return time.perf_counter() - t0, proc.stdout


def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def sample_evenly(files, cap: int) -> list:
    """Evenly spaced sample of `cap` files (stratified across the walk order),
    so the per-file mode reflects corpus-wide finding density instead of the
    (often finding-poor) first directories."""
    if cap >= len(files):
        return list(files)
    step = len(files) / cap
    return [files[int(i * step)] for i in range(cap)]


# --------------------------------------------------------------------------- #
# Participants
# --------------------------------------------------------------------------- #

def bench_linter_inproc(files) -> tuple[float, int]:
    def trial():
        n = 0
        for f in files:
            try:
                n += len(lint.analyze_file(f))
            except (SyntaxError, OSError, UnicodeDecodeError):
                pass  # same contract as the CLI: unparseable -> skipped
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


LINTER_ALL = ["--min-severity", "INFO"]  # full finding set (default hides INFO)


def bench_linter_batch(files) -> tuple[float, int]:
    def trial():
        n = 0
        for part in chunked(files, ARGV_CHUNK):
            _, out = run_cmd([sys.executable, str(LINTER), *map(str, part),
                              *LINTER_ALL, "--json"])
            n += len(json.loads(out))
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


def bench_linter_perfile(files) -> tuple[float, int]:
    def trial():
        n = 0
        for f in files:
            _, out = run_cmd([sys.executable, str(LINTER), str(f),
                              *LINTER_ALL, "--json"])
            n += len(json.loads(out))
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


def bench_hook_inproc(files) -> tuple[float, int]:
    def trial():
        n = 0
        # The hook itself is intentionally warning-agnostic; the benchmark
        # harness silences parse-time SyntaxWarning so it stays measurable
        # (subprocess participants already capture stderr).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for f in files:
                try:
                    n += len(naive_hook.findings_in_file(f))
                except (SyntaxError, OSError, UnicodeDecodeError):
                    pass  # same contract as the hook's own CLI
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


def bench_hook_batch(files) -> tuple[float, int]:
    def trial():
        n = 0
        for part in chunked(files, ARGV_CHUNK):
            _, out = run_cmd([sys.executable, str(HOOK), *map(str, part)])
            n += sum(1 for line in out.splitlines() if line.strip())
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


def bench_hook_perfile(files) -> tuple[float, int]:
    def trial():
        n = 0
        for f in files:
            _, out = run_cmd([sys.executable, str(HOOK), str(f)])
            n += sum(1 for line in out.splitlines() if line.strip())
        return n
    secs, _, n = time_median(trial, RUNS, warmup=1)
    return secs, n


def ruff_version(ruff: str) -> str:
    try:
        out = subprocess.run([ruff, "--version"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def bench_ruff_batch(files, ruff: str) -> tuple[float, int, int]:
    """Return (secs, total_diagnostics, on_ground_truth_lines)."""
    def trial():
        n_total, n_rel = 0, 0
        for part in chunked(files, ARGV_CHUNK):
            cmd = [ruff, "check", "--isolated", "--no-cache",
                   f"--select={RUFF_SELECT}", "--output-format", "json",
                   *map(str, part)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            diags = json.loads(proc.stdout) if proc.stdout.strip() else []
            n_total += len(diags)
            for d in diags:
                if TRUTH and (d["filename"], d["location"]["row"]) in TRUTH:
                    n_rel += 1
        return n_total, n_rel
    secs, _, (n_total, n_rel) = time_median(trial, RUNS, warmup=1)
    return secs, n_total, n_rel


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

RUNS = 5  # module-level, set in main() before bench_* use it
TRUTH: set[tuple[str, int]] = set()  # (filepath, lineno) ground truth lines


def fmt_ms(secs: float) -> str:
    return f"{secs * 1000:8.0f} ms"


def main(argv=None) -> int:
    global RUNS, TRUTH
    ap = argparse.ArgumentParser(
        description="Benchmark torch_hostdev_lint vs ruff vs a DIY pre-commit "
                    "hook (naive_hook.py). See module docstring.")
    ap.add_argument("--files", type=int, default=200,
                    help="synthetic corpus size (default 200)")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="scan a real directory tree instead of the synthetic "
                         "corpus (correctness table is skipped)")
    ap.add_argument("--ruff", default=None,
                    help="path to the ruff binary (default: search PATH)")
    ap.add_argument("--runs", type=int, default=5,
                    help="timed trials per mode; median is reported (default 5)")
    ap.add_argument("--per-file-cap", type=int, default=50,
                    help="max files in the per-file-subprocess mode (0 = skip "
                         "that mode; default 50)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the generated synthetic corpus (path printed)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    args = ap.parse_args(argv)
    RUNS = args.runs

    # -- corpus ------------------------------------------------------------ #
    if args.corpus:
        corpus_root, is_synthetic = args.corpus.resolve(), False
        files = lint.iter_python_files(args.corpus)
        if not files:
            print(f"no .py files under {args.corpus}", file=sys.stderr)
            return 2
        truth_counts = None
    else:
        is_synthetic = True
        corpus_root = Path(tempfile.mkdtemp(prefix="hostdev-bench-"))
        files = generate_corpus(corpus_root, args.files)
        truth_counts = {c: TRUTH_COUNTS[c] * args.files for c in TRUTH_CLASSES}
        TRUTH = {(str(f), ln) for f in files
                 for cl, ln in TRUTH_PER_FILE}
    total_bytes = sum(f.stat().st_size for f in files)

    ruff = args.ruff or shutil.which("ruff") or \
        (str(Path.home() / "miniforge3/envs/qdev/bin/ruff")
         if (Path.home() / "miniforge3/envs/qdev/bin/ruff").exists() else None)

    # -- participants -------------------------------------------------------#
    results: dict = {}

    if not args.json:  # stdout must stay pure JSON in --json mode
        print(f"corpus      : {'synthetic' if is_synthetic else str(corpus_root)}"
              f"  ({len(files)} files, {total_bytes // 1024} KiB)")
        print(f"python      : {sys.version.split()[0]}   runs: {RUNS} (median)")
        if ruff:
            print(f"ruff        : {ruff_version(ruff)}  (select {RUFF_SELECT})")
        else:
            print("ruff        : not found -- ruff participant skipped")
        if is_synthetic:
            print(f"ground truth: {truth_counts}  (per file: {TRUTH_COUNTS})")

    if args.per_file_cap > 0:
        perfile_files = sample_evenly(files, min(args.per_file_cap,
                                                  len(files)))
    else:
        perfile_files = None

    # C. this linter
    s, n = bench_linter_inproc(files)
    results["torch_hostdev_lint:in-proc"] = (s, n)
    s, n = bench_linter_batch(files)
    results["torch_hostdev_lint:batch"] = (s, n)
    if perfile_files is not None:
        s, n = bench_linter_perfile(perfile_files)
        results["torch_hostdev_lint:per-file"] = (s, n)

    # B. naive hook
    s, n = bench_hook_inproc(files)
    results["naive_hook:in-proc"] = (s, n)
    s, n = bench_hook_batch(files)
    results["naive_hook:batch"] = (s, n)
    if perfile_files is not None:
        s, n = bench_hook_perfile(perfile_files)
        results["naive_hook:per-file"] = (s, n)

    # A. ruff (batch only: it is a single binary; per-file starts are absurd)
    ruff_secs = ruff_total = ruff_rel = None
    if ruff:
        ruff_secs, ruff_total, ruff_rel = bench_ruff_batch(files, ruff)

    # -- report -------------------------------------------------------------#
    if args.json:
        out = {
            "corpus": "synthetic" if is_synthetic else str(corpus_root),
            "files": len(files), "total_bytes": total_bytes,
            "runs": RUNS, "python": sys.version.split()[0],
            "ruff_version": ruff_version(ruff) if ruff else None,
            "ground_truth": truth_counts,
            "results": {
                k: {"wall_ms": round(v[0] * 1000), "findings": v[1]}
                for k, v in results.items()
            },
            "ruff_results": {
                "wall_ms": round(ruff_secs * 1000),
                "total_diagnostics": ruff_total,
                "on_ground_truth_lines": ruff_rel,
            } if ruff_secs is not None else None,
        }
        print(json.dumps(out, indent=2))
        if is_synthetic:
            if args.keep:
                print(f"(synthetic corpus kept at {corpus_root})", file=sys.stderr)
            else:
                shutil.rmtree(corpus_root, ignore_errors=True)
        return 0

    print()
    print(f"{'participant':32s} {'wall':>10s}  findings  notes")
    print("-" * 78)
    for key, (s, n) in [
        ("ruff (batch)", (ruff_secs, ruff_total)) if ruff_secs
        else (None, None),
        *[(k, v) for k, v in results.items()],
    ]:
        if s is None:
            continue
        notes = []
        if "ruff" in key:
            if is_synthetic:
                notes.append(f"{ruff_rel} on ground-truth lines")
            else:
                notes.append("total diagnostics (no ground truth)")
        if "naive" in key:
            notes.append("HIGH-only (no .to/factories/severity/fix)")
        if "per-file" in key and perfile_files is not None:
            notes.append(f"{len(perfile_files)} evenly-sampled files")
        print(f"{key:32s} {fmt_ms(s)}  {n:>8d}  {'; '.join(notes)}")

    if is_synthetic:
        print()
        print("correctness vs ground truth:")
        l_n = results["torch_hostdev_lint:in-proc"][1]
        h_n = results["naive_hook:in-proc"][1]
        t_tot = sum(truth_counts.values())
        ok = lambda got, want: "PASS" if got == want else f"FAIL ({want})"
        print(f"  torch_hostdev_lint : {l_n:>6d} / {t_tot:>6d}   "
              f"{ok(l_n, t_tot)}")
        print(f"  naive_hook         : {h_n:>6d} / {t_tot:>6d}   "
              f"{ok(h_n, truth_counts['HIGH'])}  "
              f"(covers {100 * h_n / t_tot:.0f}% of findings; HIGH-only by design)")
        if ruff_rel is not None:
            print(f"  ruff             : {ruff_rel:>6d} on ground-truth lines   "
                  f"{ok(ruff_rel, 0)}  (built-in rules happen to touch them)")

    # takeaways
    if ruff_secs:
        print()
        print("takeaways:")
        ratio = results["torch_hostdev_lint:batch"][0] / ruff_secs
        caught = (f"it caught {ruff_rel} of the {sum(truth_counts.values())} "
                  f"ground-truth findings"
                  if is_synthetic else "no ground truth on a real corpus")
        print(f"  * ruff is {ratio:.0f}x faster than the linter (batch, "
              f"Rust vs CPython). {caught}: no built-in rule expresses "
              f"host<->device sync, and ruff has no plugin API.")
        if "torch_hostdev_lint:per-file" in results:
            pf = results["torch_hostdev_lint:per-file"]
            per = pf[0] / len(perfile_files) * 1000
            print(f"  * per-file subprocess cost: ~{per:.0f} ms/file of "
                  f"interpreter start-up. pre-commit passes all staged files in "
                  f"ONE invocation (the 'batch' row), so this only matters for "
                  f"whole-repo CI scans -- use the in-process/CI-batch form "
                  f"there.")
        if is_synthetic:
            nb = results["naive_hook:batch"][0]
            lb = results["torch_hostdev_lint:batch"][0]
            print(f"  * the DIY hook covers only "
                  f"{100 * truth_counts['HIGH'] / sum(truth_counts.values()):.0f}%"
                  f" of the finding space (no .to(device), no factories, no "
                  f"severity, no fixes); its batch run is {nb / lb:.1f}x the "
                  f"linter's time, and it is 'cheaper' only because it checks a "
                  f"fraction of the API surface.")
    if is_synthetic:
        if args.keep:
            print(f"  (synthetic corpus kept at {corpus_root})")
        else:
            shutil.rmtree(corpus_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
