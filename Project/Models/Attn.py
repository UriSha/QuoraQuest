import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class Attn(nn.Module):
    def __init__(self, embeddings_size, batch_size, cuda):
        super(Attn, self).__init__()

        self.embeddings_size = embeddings_size
        self.lin = nn.Linear(self.embeddings_size, embeddings_size)
        #         self.lin = nn.Linear(self.hidden_size*2, hidden_size*2)

        self.weight_vec = nn.Parameter(torch.FloatTensor(embeddings_size, 1))
        torch.nn.init.xavier_normal_(self.weight_vec)
        #         self.weight_vec.data[torch.isnan(self.weight_vec)] = 0
        # print()
        # print("weight_vec in init: ", self.weight_vec)
        self.batch_size = batch_size
        self.to_cuda = cuda

    # self.weight_vec = nn.Parameter(torch.FloatTensor(1, hidden_size*2))

    def forward(self, questions, questions_lens, output_log=False):

        # print()
        # print("attn.forward:")
        # print("questions.shape: ", questions.shape)

        if output_log:
            start1 = time.time()

        max_question_len = max(questions_lens)

        questions = questions.view(-1, self.embeddings_size)

        if output_log:
            print("data prep time: ", str(time.time() - start1))

        # print()
        # print("after questions.view:")
        # print("questions.shape: ", questions.shape)
        # attn_energies = torch.zeros(self.batch_size * 2, max_question_len)  # Batch_size x 1 x max_question_len
        # attn_energies = torch.zeros(questions.shape[0], self.embeddings_size)  # self.batch_size * 2 * max_question_len x self.embeddings_size


        if output_log:
            start1 = time.time()

        attn_energies = self.score(questions)

        if output_log:
            print("data prep time: ", str(time.time() - start1))

        # print()
        # print("after self.score(questions):")
        # print("attn_energies.shape: ", attn_energies.shape)

        if self.to_cuda:
            attn_energies = attn_energies.cuda()

        if output_log:
            start1 = time.time()

        attn_energies = attn_energies.view(self.batch_size * 2, max_question_len)

        if output_log:
            print("energy view time: ", str(time.time() - start1))

        # print()
        # print("after attn_energies.view:")
        # print("attn_energies.shape: ", attn_energies.shape)
        # #
        # for i in range(self.batch_size * 2):
        #     for j in range(questions_lens[i]):
        #         attn_energies[i][j] = self.score(questions[i][j])

        if output_log:
            start1 = time.time()

        masks = [self.masker(qlen, max_question_len) for qlen in questions_lens]
        masks = torch.stack(masks)
        attn_energies.masked_fill_(masks, float('-inf'))
        

        if output_log:
            print("for time: ", str(time.time() - start1))

        if output_log:
            start1 = time.time()

        # print("attn_energies before softmax: ", attn_energies)
        res = F.softmax(attn_energies, dim=1)

        if output_log:
            print("softmax time: ", str(time.time() - start1))

        # print()
        # print("after softmax:")
        # print("res.shape: ", res.shape)
        #

        return res

    def score(self, word_embed):
        #         print()
        # print()
        # print("=============================")
        # print("in score")
        # print()
        # print("word_embed.shape: ", word_embed.shape)

        energy = self.lin(word_embed)
        # print()
        # print("after self.lin")
        # print("energy.shape: ", word_embed.shape)
        # print("self.weight_vec.shape: ", self.weight_vec.shape)

        energy = torch.matmul(energy, self.weight_vec)

        # print()
        # print("after dot:")
        # print("energy.shape: ", energy.shape)
        # print("energy: ", energy)

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

    def masker(self, qlen, max_question_len):
        if qlen == max_question_len:
            return torch.zeros(max_question_len).byte()
        return torch.zeros(max_question_len).scatter_(0, torch.LongTensor(list(range(qlen, max_question_len))), 1.0).byte()