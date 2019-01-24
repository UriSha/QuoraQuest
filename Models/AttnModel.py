import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
import Attn, MasterClassifier


class AttnModel(nn.Module):
    def __init__(self, question_embed_size):
        super(AttnModel, self).__init__()

        self.question_embed_size = question_embed_size
        self.attn = Attn(question_embed_size)
        self.cls = MasterClassifier(question_embed_size)

    def forward(self, X):
        first_q = X[:self.question_embed_size]
        sec_q = X[self.question_embed_size:]

        first_weights = self.attn(first_q)
        sec_weights = self.attn(sec_q)

        first_q_avg = torch.dot(first_q, first_weights)
        sec_q_avg = torch.dot(sec_q, sec_weights)

        cls_input = np.concatenate(first_q_avg , sec_q_avg)

        result = self.cls(cls_input)

        return result

