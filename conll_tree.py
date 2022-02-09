#!/usr/bin/env python3

class Word:
    def __init__(self):
        self.wid = 0
        self.lemma = ''
        self.upos = ''
        self.feats = []
        self.head = 0
        self.rel = ''
    def from_conll(line):
        w = Word()
        parts = line.split('\t')
        w.wid = int(parts[0])
        w.lemma = parts[2]
        w.upos = parts[3]
        w.feats = parts[5].split('|')
        w.head = int(parts[6])
        w.rel = parts[7]
        return w

class Tree:
    def __init__(self):
        self.sid = ''
        self.words = []
    def from_conll(conll):
        t = Tree()
        for ln in conll.splitlines():
            if not ln.strip():
                continue
            if ln.startswith('#'):
                if 'sent_id' in ln:
                    t.sid = ln.split(' = ')[1].strip()
            else:
                if '-' in ln.split()[0]:
                    continue
                t.words.append(Word.from_conll(ln.strip()))
        return t

def iter_conll(fname):
    with open(fname) as fin:
        cur = ''
        for ln in fin:
            if not ln.strip():
                if cur:
                    yield Tree.from_conll(cur)
                cur = ''
            else:
                cur += ln
        if cur:
            yield Tree.from_conll(cur)
