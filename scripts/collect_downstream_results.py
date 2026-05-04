#!/usr/bin/env python3
import argparse, csv, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.train_jax as tj


def read_json(path):
    with tj._open_file(path, 'r') as f:
        return json.loads(f.read())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True, help='Run root, e.g. gs://.../downstream_runs/baseline_400M')
    ap.add_argument('--out', default='downstream_summary.csv')
    args=ap.parse_args()
    rows=[]
    for root in args.root:
        # roots contain task dirs; task dirs contain fixed run_name dirs or files directly.
        task_dirs = []
        try:
            # gcsfs glob only files; infer dirs from best_eval.json files.
            files = tj._list_files(root, '*/*/best_eval.json') + tj._list_files(root, '*/best_eval.json')
        except Exception:
            files = []
        for f in files:
            try:
                ev=read_json(f)
                parts=f.rstrip('/').split('/')
                # .../<task>/<run>/best_eval.json or .../<run>/best_eval.json
                task=parts[-3] if len(parts)>=3 else ''
                run=parts[-2] if len(parts)>=2 else ''
                rows.append({'root':root,'task':task,'run':run,'step':ev.get('step',''),'accuracy':ev.get('accuracy',''),'total':ev.get('total',''),'path':f})
            except Exception as e:
                print(f'WARN failed {f}: {e}', file=sys.stderr)
    with open(args.out,'w',newline='') as out:
        w=csv.DictWriter(out, fieldnames=['root','task','run','step','accuracy','total','path'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {args.out} rows={len(rows)}')
if __name__=='__main__': main()
