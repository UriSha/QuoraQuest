import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable

from Models.Attn import Attn
from Models.MasterClassifier import MasterClassifier


class AttnModel(nn.Module):
    def __init__(self, src_vecs, batch_size, cuda):
        super(AttnModel, self).__init__()

        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.w2idx = src_vecs._w2idx
        self.idx2w = src_vecs._idx2w
        self.emb_size = src_vecs.vector_size
        self.to_cuda = cuda

        # Do not update original embedding spaces
        self.semb.weight.requires_grad = False

        self.attn = Attn(self.emb_size, batch_size, cuda)
        self.cls = MasterClassifier(self.emb_size * 2)

        if self.to_cuda:
            self.cls.cuda()
            self.attn.cuda()

    def forward(self, X, lens, output_log=False):

        if output_log:
            print("torch stack")
            start = time.time()

        if self.to_cuda:
            X = torch.stack([self.semb(Variable(sent).cuda()) for sent in X])
        else:
            X = torch.stack([self.semb(Variable(sent)) for sent in X])

        if output_log:
            end = time.time()
            print("torch stack ended: ", str(end - start))

        if output_log:
            print("Start forward pass of attention model")
            start = time.time()
        weights = self.attn(X, lens, output_log)

        if output_log:
            end = time.time()
            print("Attention model forward pass ended after: ", str(end - start))
        # print()
        # print("in attn model after self.attn()")
        # print("weights.shape", weights.shape)
        # print("X.shape", X.shape)
        # X = torch.stack(X)
        #         print("X: ", X)
        #         print("X.shape: ", X.shape)

        #         print("weights: ", weights)
        #         print("weights.shape: ", weights.shape)


        # X = torch.dot(X, weights)
        if output_log:
            print("Start weight manipulations")
            start = time.time()

        weights = weights.unsqueeze(2)
        weights = weights.expand(weights.shape[0], weights.shape[1], self.emb_size)
        weigthed_outputs = torch.mul(X, weights)
        X = torch.sum(weigthed_outputs, -2)

        if output_log:
            end = time.time()
            print("Weight manipulation ended after: ", str(end - start))

        if output_log:
            print("Start master classifier")
            start = time.time()

        X = self.concat_pairs(X)
        X = self.cls(X)
        X = X.squeeze(dim=1)

        if output_log:
            end = time.time()
            print("Master classifier ended after: ", str(end - start))

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
