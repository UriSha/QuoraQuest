import numpy as np
import torch
import torch.nn as nn

from Models.Attn import Attn
from Models.MasterClassifier import MasterClassifier


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

        cls_input = np.concatenate(first_q_avg, sec_q_avg)

        result = self.cls(cls_input)

        return result

    def prepare_batch(self, batch_x):
        # get the length of each sentence
        batch_lengths = [len(sentence) for sentence in batch_x]

        batch_x = [self.sent_to_indecies(sent) for sent in batch_x]
        # create an empty matrix with padding tokens
        #   pad_token = vocab['<PAD>']
        pad_token = 0
        longest_sent = max(batch_lengths)
        batch_size = len(batch_x)
        padded_batch = np.ones((batch_size, longest_sent)) * pad_token
        # copy over the actual sequences
        for i, sent_len in enumerate(batch_lengths):
            sequence = batch_x[i]
            padded_batch[i, 0:sent_len] = sequence[:sent_len]

        batch = [torch.cuda.LongTensor(l) for l in padded_batch]
        return torch.stack(batch), batch_lengths

    def sent_to_indecies(self, sentence):
        return [self.w2idx(word) for word in sentence]

