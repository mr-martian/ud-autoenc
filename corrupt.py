#!/usr/bin/env python3

import conll_tree as CT
import random
import copy

ALL_CORRUPTERS = {}

def list_corrupters():
    return list(ALL_CORRUPTERS.keys())

def corrupt(tree, cor_list, cor_count=1):
    keys = []
    if 'all' in cor_list:
        keys = list(ALL_CORRUPTERS.keys())
    else:
        keys = [c for c in cor_list if c in ALL_CORRUPTERS]
    if not keys:
        raise RuntimeError('corrupt(%s) includes no valid corruptions!' % cor_list)
    ret = copy.deepcopy(tree)
    for i in range(cor_count):
        k = random.choice(keys)
        ALL_CORRUPTERS[k](ret)
    return ret

def corrupter(func):
    global ALL_CORRUPTERS
    ALL_CORRUPTERS[func.__name__] = func
    return func

@corrupter
def delfn(tree):
    dl = []
    for i, w in enumerate(tree.words):
        if w.upos in ['ADP', 'AUX'] and len(list(tree.children(w.wid))) == 0:
            dl.append(i)
    for idx in reversed(dl):
        del tree.words[idx]
    tree.update_index()
    return tree
