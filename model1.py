#!/usr/bin/env python3

from model_base import *

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
