#!/usr/bin/env python3
"""Collect downstream best/final eval JSON files into a CSV/Markdown table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root output dir, e.g. runs/downstream")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows: List[Dict] = []
    for run_dir in sorted(root.glob("**")):
        if not run_dir.is_dir():
            continue
        cfg = read_json(run_dir / "run_config.json")
        best = read_json(run_dir / "best_eval.json")
        final = read_json(run_dir / "final_eval.json")
        if not cfg and not best and not final:
            continue
        rows.append({
            "model_type": cfg.get("model_type", run_dir.parts[-2] if len(run_dir.parts) >= 2 else ""),
            "task": cfg.get("task", run_dir.name),
            "checkpoint": cfg.get("checkpoint", ""),
            "best_acc": best.get("accuracy", ""),
            "best_step": best.get("step", ""),
            "final_acc": final.get("accuracy", ""),
            "final_step": final.get("step", ""),
            "total_eval": best.get("total", final.get("total", "")),
            "run_dir": str(run_dir),
        })

    rows.sort(key=lambda r: (str(r["task"]), str(r["model_type"]), str(r["run_dir"])))

    out_csv = Path(args.out_csv) if args.out_csv else root / "downstream_results.csv"
    out_md = Path(args.out_md) if args.out_md else root / "downstream_results.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task", "model_type", "best_acc", "best_step", "final_acc", "final_step", "total_eval", "checkpoint", "run_dir"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    lines = ["| task | model | best acc | best step | final acc | final step | n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)
        lines.append(
            f"| {r['task']} | {r['model_type']} | {fmt(r['best_acc'])} | {r['best_step']} | "
            f"{fmt(r['final_acc'])} | {r['final_step']} | {r['total_eval']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
