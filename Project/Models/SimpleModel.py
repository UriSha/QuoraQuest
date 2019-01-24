import torch.nn as nn
import torch
import numpy as np
from MasterClassifier import MasterClassifier
from torch.autograd import Variable


class SimpleModel(nn.Module):
    def __init__(self, src_vecs):
        super(SimpleModel, self).__init__()

        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.sw2idx = src_vecs._w2idx
        self.sidx2w = src_vecs._idx2w
        self.emb_size = src_vecs.vector_size

        # Do not update original embedding spaces
        self.semb.weight.requires_grad = False

        self.cls = MasterClassifier.MasterClassifier(src_vecs.vector_size * 2)

    def forward(self, X):

        X = self.ave_vecs(X)

        X = self.concat_pairs(X)

        return self.cls(X)

    def concat_pairs(self, X):
        """
        Gets a batch of embedded pairs of questions
         (a pair of questions is the i'th tensor and (i+1)'th tensor for every even i)
        @:return Tensor of concatenated pairs of questions
        """
        return torch.stack([torch.cat((X[i], X[i + 1]), 0) for i in range(0, len(X), 2)])

    def idx_vecs(self, sentence, model):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        """
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except KeyError:
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        """
        Converts a batch of tokenized sentences
        to a matrix of word indices from model.
        """

        return [self.idx_vecs(s, model) for s in X]

    def ave_vecs(self, X):
        """
        Converts a batch of tokenized sentences into
        a matrix of averaged word embeddings. If src
        is True, it uses the sw2idx and semb to create
        the averaged representation. Otherwise, it uses
        the tw2idx and temb.
        """
        vecs = []

        idxs = self.lookup(X, self.sw2idx)
        for i in idxs:
            to_add = self.semb(Variable(i)).mean(0)
            vecs.append(to_add)

        return torch.stack(vecs)
