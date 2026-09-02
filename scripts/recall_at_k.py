import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("decls"); ap.add_argument("retrieved")
ap.add_argument("--k", type=int, nargs="+", default=[16, 32])
a = ap.parse_args()

name2rel = {d["decl_name"]: set(d["gt_premises"]) for d in json.load(open(a.decls))}
entries = json.load(open(a.retrieved))
if isinstance(entries, dict):
    entries = entries["dot"]

for k in a.k:
    recalls, precs, fulls, missing = [], [], [], 0
    for e in entries:
        rel = name2rel.get(e["decl_name"])
        if rel is None:
            missing += 1; continue
        top = {p["corpus_id"] for p in sorted(e["premises"], key=lambda p: p["score"], reverse=True)[:k]}
        tp = rel & top
        recalls.append(len(tp) / len(rel) if rel else 1.0)
        precs.append(len(tp) / k)
        fulls.append(rel <= top)
    print(f"k={k:3d}  n={len(recalls)}  missing={missing}  Recall@k={100*np.mean(recalls):.2f}%  "
          f"Precision@k={100*np.mean(precs):.2f}%  FullRecall@k={100*np.mean(fulls):.2f}%")
