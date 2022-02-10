#!/usr/bin/env python3

from model_base import *
from conll_tree import Tree

class Model1(ModelBase):
    def __init__(self, formatter, state_size):
        super().__init__()
        self.form = formatter
        self.lstm = nn.LSTM(input_size=self.form.dim, hidden_size=state_size,
                            num_layers=1, batch_first=True)
        self.linear = nn.Linear(state_size, self.form.dim)
        self.relu = nn.ReLU()
    def forward(self, vec):
        i1 = vec.type(torch.FloatTensor).unsqueeze(0)
        i2 = self.lstm(i1)[0]
        i3 = self.linear(i2)
        return self.relu(i3).squeeze(0)[-1]
    def single_step(self, vec, state):
        i0 = vec.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        i1, h = self.lstm(i0, state) if state != None else self.lstm(i0)
        i2 = self.linear(i1)
        i3 = self.relu(i2).squeeze(0).squeeze(0)
        return i3, h
    def predict(self, tree):
        pred = None
        hid = None
        for v in self.form.to_list(tree):
            pred, hid = self.single_step(v, hid)
        otree = Tree()
        otree.sid = tree.sid
        while not self.form.is_eos(pred):
            self.form.add_word(pred, otree)
            if len(otree.words) > len(tree.words)*5:
                break
            pred, hid = self.single_step(pred, hid)
        return otree
