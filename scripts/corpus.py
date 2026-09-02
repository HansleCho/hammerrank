import os, pickle


class Corpus:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("this release loads the prebuilt corpus cache; run scripts/download_artifacts.sh")

    def accessible(self, module, line, column, include_self_module=True):
        mods = set(self.imports.get(module, set())); mods.discard(module)
        acc = set()
        for m in mods:
            acc.update(self.module_premises.get(m, ()))
        if include_self_module:
            for n in self.module_premises.get(module, ()):
                p = self.premises[n]
                if (p["line"], p["column"]) <= (line, column):
                    acc.add(n)
        return acc


def load(cache="data/corpus_v4.18.0.pkl"):
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} not found; run scripts/download_artifacts.sh first")
    return pickle.load(open(cache, "rb"))
