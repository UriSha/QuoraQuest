import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn(nn.Module):
    def __init__(self, embeddings_size, batch_size, cuda):
        super(Attn, self).__init__()

        self.embeddings_size = embeddings_size
        self.lin = nn.Linear(self.embeddings_size, embeddings_size)
        #         self.lin = nn.Linear(self.hidden_size*2, hidden_size*2)

        self.weight_vec = nn.Parameter(torch.FloatTensor(1, embeddings_size))
        torch.nn.init.xavier_normal_(self.weight_vec)
        #         self.weight_vec.data[torch.isnan(self.weight_vec)] = 0
        # print()
        # print("weight_vec in init: ", self.weight_vec)
        self.batch_size = batch_size
        self.to_cuda = cuda

    # self.weight_vec = nn.Parameter(torch.FloatTensor(1, hidden_size*2))

    def forward(self, questions, questions_lens):

        questions_lens = np.array([l for l in questions_lens])
        max_question_len = np.max(questions_lens)

        attn_energies = torch.zeros(self.batch_size * 2, max_question_len)  # Batch_size x 1 x max_question_len

        if self.to_cuda:
            attn_energies = attn_energies.cuda()

        for i in range(self.batch_size * 2):
            for j in range(questions_lens[i]):
                attn_energies[i][j] = self.score(questions[i][j])

        for i in range(self.batch_size * 2):
            for j in range(questions_lens[i], max_question_len):
                attn_energies[i][j] = float('-inf')

        # print("attn_energies before softmax: ", attn_energies)
        res = F.softmax(attn_energies, dim=1)

        return res

    def score(self, word_embed):
        #         print()
        #         print()
        #         print("=============================")
        #         print()
        #         print()
        # print("word_embed.shape: ", word_embed.shape)
        # print("word_embed: ", word_embed)

        energy = self.lin(word_embed)
        #         print()
        #         print()
        #         print("weight_vec in score: ",self.weight_vec)
        energy = torch.dot(self.weight_vec.view(-1), energy)

        #         print()

        #         print("after dot:" )
        #         print("energy.shape: ", energy.shape)
        #         print("energy: ", energy)

        return energy

        # def forward(self, question):
        #     seq_len = len(question)
        #
        #     attn_energies = torch.zeros(seq_len).cuda()  # B x 1 x S
        #
        #     for i in range(seq_len):
        #         attn_energies[i] = self.score(question[i])
        #
        #     # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        #     return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
        #
        # def score(self, word):
        #     energy = self.lin(word)
        #     energy = torch.dot(self.weight_vec.view(-1), energy.view(-1))
        #     return energy
