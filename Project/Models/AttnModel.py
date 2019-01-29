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

    def forward(self, X, lens):

        weights = self.attn(X, lens)
        
        #X = torch.dot(X, weights)
        weights = weights.unsqueeze(2)
        weights = weights.expand(weights.shape[0],weights.shape[1],self.emb_size)
        weigthed_outputs = torch.mul(X, weights)
        X = torch.sum(weigthed_outputs, -2)

        X = self.concat_pairs(X)
        X = self.cls(X)
        X = X.squeeze(dim=1)

        return X

    def idx_vecs(self, sentence):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        """
        if len(sentence) == 0:
            sentence = ["NA"]
        sent = []
        for w in sentence:
            try:
                sent.append(self.w2idx[w])
            except KeyError:
                sent.append(0)

        if self.to_cuda:
            return torch.cuda.LongTensor(np.array(sent))

        return torch.LongTensor(np.array(sent))

    def concat_pairs(self, X):
        """
        Gets a batch of embedded pairs of questions
         (a pair of questions is the i'th tensor and (i+1)'th tensor for every even i)
        @:return Tensor of concatenated pairs of questions
        """
        return torch.stack([torch.cat((X[i], X[i + 1]), 0) for i in range(0, len(X), 2)])
