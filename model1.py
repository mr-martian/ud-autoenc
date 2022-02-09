#!/usr/bin/env python3

from model_base import *

class Model1(ModelBase):
    def __init__(self, formatter, state_size):
        super().__init__()
        self.form = formatter
        self.lstm = nn.LSTM(input_size=self.form.dim, hidden_size=state_size,
                            num_layers=1)
        self.linear = nn.Linear(state_size, self.form.dim)
        self.relu = nn.ReLU()
    def forward(self, vec):
        i1 = self.lstm(vec)[0]
        i2 = self.linear(i1)
        return self.relu(i2)
