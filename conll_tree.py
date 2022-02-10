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
        if parts[5] != '_':
            w.feats = parts[5].split('|')
        w.head = int(parts[6])
        w.rel = parts[7]
        return w
    def to_conll(self):
        ls = [
            str(self.wid),
            '_',
            self.lemma or '_',
            self.upos,
            '_',
            '|'.join(sorted(self.feats)),
            str(self.head),
            self.rel,
            '_',
            '_'
        ]
        return '\t'.join(ls)

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
    def to_conll(self):
        ret = ['# sent_id = ' + self.sid] + [w.to_conll() for w in self.words]
        return '\n'.join(ret) + '\n\n'
    def update_index(self):
        upd = {}
        for i, w in enumerate(self.words, 1):
            upd[w.wid] = i
        for w in self.words:
            w.wid = upd[w.wid]
            w.head = upd.get(w.head, 0)
    def children(self, wid):
        for w in self.words:
            if w.head == wid:
                yield w

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
