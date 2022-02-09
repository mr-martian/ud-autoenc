#!/usr/bin/env python3

from format_base import *

def one_hot(opts, val, unk=-1):
    ret = [0]*len(opts)
    i = opts.index(val)
    if i == -1:
        i = unk
    ret[i] = 1
    return ret

class Format1(FormatBase):
    def __init__(self, lemma, feat, rel):
        self.lemma = lemma
        self.pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.feat = feat
        self.rel = rel
        self.dim = 3 + len(lemma) + len(self.pos) + len(feat) + len(rel)
        # [ EOS, left, dist, UNK, lem..., POS..., feat..., rel...]
    def from_corpus(corp):
        l = set([''])
        f = set()
        r = set(['dep'])
        for t in corp:
            for w in t.words:
                l.add(w.lemma)
                f.update(w.feats)
                r.add(w.rel)
        return Format1(sorted(l), sorted(f), sorted(r))
    def eos(self):
        arr = [0] * self.dim
        arr[0] = 1
        return torch.Tensor(arr).type(torch.LongTensor)
    def word_to_vec(self, w):
        arr = [0]
        if w.head == 0:
            arr += [0, 0]
        elif w.head < w.wid:
            arr += [1, w.wid - w.head]
        else:
            arr += [0, w.head - w.wid]
        arr += one_hot(self.lemma, w.lemma, 0)
        arr += one_hot(self.pos, w.upos, -1)
        feat = [0]*len(self.feat)
        for f in w.feats:
            i = self.feat.index(f)
            if i != -1:
                feat[i] = 1
        arr += feat
        arr += one_hot(self.rel, w.rel, 0)
        return torch.Tensor(arr).type(torch.LongTensor)
