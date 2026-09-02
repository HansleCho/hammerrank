import argparse, json, math, os, pickle, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import load as load_corpus

TOK = re.compile(r"[A-Za-z_\u00c0-\u02af\u0370-\u03ff\u2100-\u214f][A-Za-z0-9_.'\u00c0-\u02af\u0370-\u03ff\u2070-\u209f\u2100-\u214f!?\u2080-\u2089]*")

def weight(freq_t):
    return 1.0 + 2.0 / (float(max(freq_t, 1).bit_length() - 1) + 1.0)

def build_index(corpus, tt):
    names = list(corpus.premises.keys())
    row = {n: i for i, n in enumerate(names)}
    N = len(names)
    ptoks = [tt["prem_toks"][n] for n in names]
    plen = np.array([max(len(t), 1) for t in ptoks], dtype=np.float32)
    df = tt["freq"]
    idf = {t: math.log(N / (1 + d)) for t, d in df.items()}
    inv = {}
    for i, ts in enumerate(ptoks):
        for t in ts: inv.setdefault(t, []).append(i)
    inv = {t: np.array(v, dtype=np.int32) for t, v in inv.items()}
    ranges = {}
    for i, n in enumerate(names):
        m = corpus.premises[n]["module"]
        a, b = ranges.get(m, (i, i))
        ranges[m] = (min(a, i), max(b, i))
    known = tt["tok2id"]
    type_toks, heads = [], []
    for n in names:
        tl = [t for t in TOK.findall(corpus.premises[n]["type"] or "") if t in known]
        type_toks.append(frozenset(known[t] for t in tl)); heads.append(tl[0] if tl else "")
    return dict(names=names, row=row, N=N, ptoks=ptoks, plen=plen, idf=idf, inv=inv, ranges=ranges,
                type_toks=type_toks, heads=heads)

def module_mask(corpus, ix, module, cache={}):
    if module in cache: return cache[module]
    mask = np.zeros(ix["N"], dtype=bool)
    for m in corpus.imports.get(module, ()):
        if m != module and m in ix["ranges"]:
            a, b = ix["ranges"][m]; mask[a:b+1] = True
    if len(cache) > 400: cache.clear()
    cache[module] = mask
    return mask

def build_knn(hd, train_names, idf):
    inv = {}
    for j, dn in enumerate(train_names):
        for t in hd[dn]["toks"]: inv.setdefault(t, []).append(j)
    inv = {t: np.array(v, dtype=np.int32) for t, v in inv.items()}
    norms = np.array([math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in hd[dn]["toks"])) or 1.0 for dn in train_names],
                     dtype=np.float32)
    return dict(inv=inv, norms=norms, names=train_names, pos={n: j for j, n in enumerate(train_names)})

def knn_votes(knn, hd, gtoks, idf, k=128, pw=4, exclude=None):
    sc = np.zeros(len(knn["names"]), dtype=np.float32)
    for t in gtoks:
        if t in knn["inv"]: sc[knn["inv"][t]] += idf.get(t, 0.0) ** 2
    sc = sc / knn["norms"]
    if exclude is not None and exclude in knn.get("pos", {}): sc[knn["pos"][exclude]] = -1.0 
    kk = min(k, len(sc))
    top = np.argpartition(-sc, kk - 1)[:kk]
    votes = {}
    for j in top:
        w = float(sc[j]) ** pw
        if w <= 0: continue
        for g in hd[knn["names"][j]]["gt"]:
            votes[g] = votes.get(g, 0.0) + w
    return votes

def decl_features(corpus, tt, ix, dn, dinfo, ncand, prior, knn=None, hd=None, cooc=None):
    gtoks = dinfo["toks"]
    if not gtoks: return None
    module = dinfo["module"]
    mask = module_mask(corpus, ix, module).copy()
    if module in ix["ranges"]:
        a, b = ix["ranges"][module]
        for i in range(a, b + 1):
            p = corpus.premises[ix["names"][i]]
            if (p["line"], p["column"]) < (dinfo["line"], dinfo["column"]): mask[i] = True
    if dn in ix["row"]: mask[ix["row"][dn]] = False
    idf, inv = ix["idf"], ix["inv"]
    sc = np.zeros(ix["N"], dtype=np.float32)
    gnorm = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in gtoks)) or 1.0
    for t in gtoks:
        if t in inv: sc[inv[t]] += idf[t] ** 2
    tfidf = sc / (np.sqrt(ix["plen"]) * gnorm)
    tfidf[~mask] = -1.0
    k = min(ncand, int(mask.sum()))
    if k <= 0: return None
    top = np.argpartition(-tfidf, k - 1)[:k]
    top = top[np.argsort(-tfidf[top])]
    votes = knn_votes(knn, hd, gtoks, ix["idf"], exclude=dn) if knn is not None else {}
    extra = []
    extra += [ix["row"][n] for n, v in sorted(votes.items(), key=lambda kv: -kv[1])[:ncand] if n in ix["row"]]
    extra += [ix["row"][n] for n, c in sorted(prior.items(), key=lambda kv: -kv[1])[:128] if n in ix["row"]]
    if module in ix["ranges"]:
        a2, b2 = ix["ranges"][module]
        extra += list(range(a2, b2 + 1))
    extra = [i for i in extra if mask[i]]
    top = np.unique(np.concatenate([top, np.array(extra, dtype=top.dtype)])) if extra else top
    gset = set(gtoks); gt_set = set(dinfo["gt"])
    goal_only = dinfo.get("goal_toks", gset); head = dinfo.get("head", "")
    aux_on = "static" in ix
    if aux_on:
        gns = set()
        for t in gset:
            nm = ix["id2tok"].get(t, "")
            parts = nm.split("."); gns.update(".".join(parts[:i]) for i in range(1, len(parts)))
        garea = module.split(".")[1] if module.startswith("Mathlib.") else ""
        gdepth = max(1, len(corpus.imports.get(module, ())))
        pd_off, pd_used = set(), set()
    vrank = {n: r for r, (n, v) in enumerate(sorted(votes.items(), key=lambda kv: -kv[1]))}
    topq = sorted(votes.items(), key=lambda kv: -kv[1])[:16]
    cnt = cooc["cnt"] if cooc else {}; cc = cooc["cooc"] if cooc else {}
    W = weight
    rows = []
    for i in top:
        ts = ix["ptoks"][i]; inter = gset & ts
        ov = len(inter)
        wcov = sum(W(tt["freq"].get(t, 0)) for t in inter)
        pl = len(ts) or 1
        name = ix["names"][i]; p = corpus.premises[name]
        mepo = (sum(W(tt["freq"].get(t, 0)) for t in inter) /
                (sum(W(tt["freq"].get(t, 0)) for t in inter) + (pl - ov))) if (ov or pl) else 0.0
        dncomp = set(dn.split(".")); pcomp = set(name.split("."))
        tts = ix["type_toks"][i]
        cov_type = len(tts & goal_only) / max(len(tts), 1)
        co = sum(v * cc.get(name, {}).get(q, 0) / max(cnt.get(q, 1), 1) for q, v in topq) if cc else 0.0
        if aux_on:
            st = ix["static"][i]
            off = ix["offered"].get(name, 0) - (1 if name in pd_off else 0)
            usd = ix["used"].get(name, 0) - (1 if name in pd_used else 0)
            extra_f = (float(st[0]), float(st[1]), float(st[2]), float(st[3]), float(st[4]), float(st[5]), float(st[6]), float(st[7]), float(st[8]),
                       float(len(ix["ns"][i] & gns)), float(len(ix["ns"][i] & gns) / max(1, len(ix["ns"][i]))), float(st[9] == garea),
                       float(st[10] / gdepth), float(off), float(usd), float((usd + 0.5) / (off + 2.0)))
        else:
            extra_f = (0.0,) * 16
        rows.append((dn, name, *extra_f, float(votes.get(name, 0.0)),
            float(vrank.get(name, 1e6)),
            float(co), float(cov_type), float(ix["heads"][i] == head),
            float(tfidf[i]), float(ov), float(ov / pl), float(ov / max(len(gset), 1)), float(wcov), float(mepo),
            float(len(gset)), float(pl), float(p["module"] == module),
            float(abs(p["line"] - dinfo["line"])) if p["module"] == module else 1e6,
            float(prior.get(name, 0)), float(p["kind"] == "theorem"), float(p["isProp"]),
            float(len(dncomp & pcomp)), float(p["doc"] is not None),
            int(name in gt_set)))
    return rows

AUX = ["n_args","n_inst","n_forall","n_exists","n_lambda","n_fun_hyp","n_type_args","type_len","n_syms",
       "ns_overlap","ns_frac","same_area","depth_ratio","atp_offered","atp_used","atp_rate"]
COLS = ["decl_name","premise"] + AUX + ["knn","knn_grank","cooc","cov_type","head_match","tfidf","overlap","cov_p","cov_g","wcov","mepo","glen","plen",
        "same_mod","line_dist","prior","is_thm","is_prop","name_sim","has_doc","label"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decls", required=True, help='JSON list of {"decl_name": ..., "gt_premises": [...]} (gt_premises may be empty)')
    ap.add_argument("--out", required=True, help="output parquet for hr_score.py")
    ap.add_argument("--ncand", type=int, default=256, help="candidate width per generator; the released model uses 256")
    a = ap.parse_args()
    corpus = load_corpus(); hd = pickle.load(open("data/hr_data.pkl", "rb"))
    tt = pickle.load(open("data/symsel_tokens_exact.pkl", "rb"))
    goal_exact = pickle.load(open("data/hr_goal_exact.pkl", "rb"))
    for dn, d in hd.items():  
        g = goal_exact.get(dn)
        if g: d["toks"] = g; d["goal_toks"] = g
    print("using exact symbols:", len(goal_exact), "goals")
    ix = build_index(corpus, tt)
    held = set()
    for f in ["test_decls.json", "valid_decls.json", "ldtest_decls.json", "ldval_decls.json"]:
        pth = os.path.join("data/splits", f)
        if os.path.exists(pth): held.update(d["decl_name"] for d in json.load(open(pth)))
    train_names = sorted(dn for dn, d in hd.items() if d["gt"] and dn not in held and d["in_corpus"])
    prior = {}
    for dn in train_names:
        for g in hd[dn]["gt"]: prior[g] = prior.get(g, 0) + 1
    aux = pickle.load(open("data/hr_aux_features.pkl", "rb"))
    ix["static"] = [aux["static"].get(n, (0,0,0,0,0,0,0,0,0,"",0)) for n in ix["names"]]
    ix["offered"], ix["used"], ix["per_decl"] = aux["offered"], aux["used"], aux["per_decl"]
    id2tok = {i: t for t, i in tt["tok2id"].items()}
    ix["ns"] = [set(".".join(n.split(".")[:i]) for i in range(1, len(n.split(".")))) for n in ix["names"]]
    ix["id2tok"] = id2tok
    print("aux features loaded")
    knn = build_knn(hd, train_names, ix["idf"])
    cooc = pickle.load(open("data/hr_cooc.pkl", "rb"))
    t0 = time.time()
    decls = json.load(open(a.decls))
    buf = []
    for d in decls:
        dn = d["decl_name"]
        if dn not in hd: print("no data for", dn); continue
        r = decl_features(corpus, tt, ix, dn, hd[dn], a.ncand, prior, knn=knn, hd=hd, cooc=cooc)
        if r: buf.extend(r)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(buf, columns=COLS).to_parquet(a.out)
    print(f"features: {len(decls)} decls -> {a.out} ({len(buf)} rows, {(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
