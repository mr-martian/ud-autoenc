#!/usr/bin/env python3

from format_base import *

class Format1(FormatBase):
    def __init__(self, lemma, pos, feat, rel):
        self.lemma = lemma
        self.pos = pos
        self.feat = feat
        self.rel = rel
        self.dim = 7 + len(lemma) + len(pos) + len(feat) + len(rel)
    def from_corpus(corp):
        l = set()
        f = set()
        r = set()
        for t in corp:
            for w in t.words:
                l.add(w.lemma)
                f.update(w.feats)
                r.add(w.rel)
        fm = Format1()
        fm.lemmas = sorted(l)
        fm.feats = sorted(f)
        fm.rels = sorted(r)
        return fm
