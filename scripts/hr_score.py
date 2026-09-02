import sys, json, os, pandas as pd, lightgbm as lgb
bst = lgb.Booster(model_file=sys.argv[1]); df = pd.read_parquet(sys.argv[2])
df["score"] = bst.predict(df[bst.feature_name()], num_iteration=bst.best_iteration or None)
out = []
for dn, g in df.groupby("decl_name"):
    top = g.nlargest(1024, "score")
    out.append({"decl_name": dn, "premises": [{"corpus_id": p, "score": float(s)} for p, s in zip(top["premise"], top["score"])]})
os.makedirs(os.path.dirname(sys.argv[3]) or ".", exist_ok=True); json.dump(out, open(sys.argv[3], "w")); print(len(out), "rows ->", sys.argv[3])
