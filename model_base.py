#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelBase(nn.Module):
    def full_pipeline(self, tree):
        vec = self.form.to_vec(tree).to(device)
        out = self.forward(vec)
        return self.form.to_tree(out)

class Trainer():
    def __init__(self, model, lr=0.001):
        self.model = model
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=lr)
        self.losses = []
    def run_sent(self, in_sent, out_sent):
        in_vec = self.model.form.to_list(in_sent)
        out_vec = self.model.form.to_list(out_sent)
        for i in range(len(out_vec)):
            test = torch.stack(in_vec + out_vec[:i]).type(torch.FloatTensor).to(device)
            out = self.model.forward(test)
            loss = F.mse_loss(out, out_vec[i].type(torch.FloatTensor))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
    def epoch(self, sents):
        x = len(self.losses)
        for si, so in sents:
            self.run_sent(si, so)
        print(np.array(self.losses[x:]).mean())
