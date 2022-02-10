#!/usr/bin/env python3

import argparse
import torch

import conll_tree as CT
from model_base import Trainer
from corrupt import corrupt, list_corrupters

import format1

import model1

parser = argparse.ArgumentParser()
parser.add_argument('model', action='store')
parser.add_argument('infile', action='store')
parser.add_argument('outfile', action='store')
args = parser.parse_args()

model = torch.load(args.model)
model.eval()

with open(args.outfile, 'w') as fout:
    for tree in CT.iter_conll(args.infile):
        otree = model.predict(tree)
        fout.write(otree.to_conll())

