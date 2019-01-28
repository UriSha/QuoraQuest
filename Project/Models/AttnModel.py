import numpy as np
import torch
import torch.nn as nn

from Models.Attn import Attn
from Models.MasterClassifier import MasterClassifier


class AttnModel(nn.Module):
    def __init__(self, src_vecs, cuda):
        super(AttnModel, self).__init__()

        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.w2idx = src_vecs._w2idx
        self.idx2w = src_vecs._idx2w
        self.emb_size = src_vecs.vector_size
        self.to_cuda = cuda

        # Do not update original embedding spaces
        self.semb.weight.requires_grad = False

        self.attn = Attn(self.emb_size)
        self.cls = MasterClassifier(self.emb_size * 2)

        if self.to_cuda:
            self.cls.cuda()
            self.attn.cuda()

    def forward(self, X):

        #X = Padder(X)

        X = self.concat_pairs(X)

        X = self.cls(X)

        X = X.squeeze(dim=1)

        return X


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

        batch_x = [self.sent_to_indices(sent) for sent in batch_x]
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

        if self.to_cuda:
            batch = [torch.LongTensor(l) for l in padded_batch]
        else:
            batch = [torch.cuda.LongTensor(l) for l in padded_batch]
        return torch.stack(batch), batch_lengths

    def sent_to_indices(self, sentence):
        return [self.w2idx(word) for word in sentence]

    def concat_pairs(self, X):
        """
        Gets a batch of embedded pairs of questions
         (a pair of questions is the i'th tensor and (i+1)'th tensor for every even i)
        @:return Tensor of concatenated pairs of questions
        """
        return torch.stack([torch.cat((X[i], X[i + 1]), 0) for i in range(0, len(X), 2)])
