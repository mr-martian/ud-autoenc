#!/usr/bin/env python3

import argparse
import torch

import conll_tree as CT
from model_base import Trainer

import format1

import model1

formats = [format1.Format1]
models = [model1.Model1]

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', action='store', type=int,
                    choices=range(1,len(models)+1), default=len(models))
parser.add_argument('-f', '--format', action='store', type=int,
                    choices=range(1,len(formats)+1), default=len(models))
parser.add_argument('-e', '--epochs', action='store', type=int,
                    default=20)
parser.add_argument('-s', '--size', action='store', type=int,
                    default=100, help='size of hidden state')
parser.add_argument('corpus', action='store')
parser.add_argument('out', action='store')
args = parser.parse_args()

trees = list(CT.iter_conll(args.corpus))

form = formats[args.format-1].from_corpus(trees)
model = models[args.model-1](form, args.size)

train = Trainer(model)
train_data = [(t,t) for t in trees] # TODO: corruptions
for i in range(1, args.epochs+1):
    print('epoch', i)
    train.epoch(train_data)

torch.save(model, args.out)
