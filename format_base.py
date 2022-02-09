#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import conll_tree as CT

class FormatBase:
    dim = 1
    def to_vec(self, tree):
        return torch.Tensor(self.to_list(tree)).type(torch.LongTensor)
    def to_list(self, tree):
        return [self.word_to_vec(w) for w in tree.words] + [self.eos()]
