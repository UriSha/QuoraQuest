import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, embeddings_size, batch_size, cuda):
        super(Attn, self).__init__()

        self.embeddings_size = embeddings_size
        self.lin = nn.Linear(self.embeddings_size, embeddings_size)

        self.weight_vec = nn.Parameter(torch.FloatTensor(embeddings_size, 1))
        torch.nn.init.xavier_normal_(self.weight_vec)

        self.batch_size = batch_size
        self.to_cuda = cuda


    def forward(self, questions, questions_lens, output_log=False):
        max_question_len = max(questions_lens)
        questions = questions.view(-1, self.embeddings_size)

        attn_energies = self.score(questions)

        if self.to_cuda:
            attn_energies = attn_energies.cuda()

        attn_energies = attn_energies.view(self.batch_size * 2, max_question_len)

        masks = [self.masker(qlen, max_question_len) for qlen in questions_lens]
        masks = torch.stack(masks)

        if self.to_cuda:
            masks = masks.cuda()
        attn_energies.masked_fill_(masks, float('-inf'))

        res = F.softmax(attn_energies, dim=1)
        return res

    def score(self, word_embed):
        energy = self.lin(word_embed)
        energy = torch.matmul(energy, self.weight_vec)
        return energy

    def masker(self, qlen, max_question_len):
        if qlen == max_question_len:
            if self.to_cuda:
                return torch.zeros(max_question_len).byte().cuda()
            return torch.zeros(max_question_len).byte()
        if self.to_cuda:
            return torch.zeros(max_question_len).scatter_(0, torch.LongTensor(list(range(qlen, max_question_len))), 1.0).byte().cuda()    
        return torch.zeros(max_question_len).scatter_(0, torch.LongTensor(list(range(qlen, max_question_len))), 1.0).byte()