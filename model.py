import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.seq_1 = nn.GRUCell(embedding_size, hidden_size)
        self.seq_2 = nn.GRUCell(hidden_size, hidden_size)
        self.seq_3 = nn.GRUCell(hidden_size, hidden_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        x = self.embedding(x)
        h_out = torch.zeros(h.size(), requires_grad=True, device=self.device)
        x = h_out[0] = self.seq_1(x, h[0])
        x = h_out[1] = self.seq_1(x, h[1])
        x = h_out[2] = self.seq_1(x, h[2])

        x = self.linear(x)

        return x, h_out

    def init_hidden(self, batch_size):
        return torch.randn(3, batch_size, 512, requires_grad=True, device=self.device)


class RL():
    def __init__(self, voc, embed_size=128, hidden_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq = GRU(voc.vocab_size, embed_size, hidden_size).to(self.device)
        
        self.voc = voc
        
    def probs(self, target: torch.Tensor):
        batch_size, seq_length = target.size()
        start_token = torch.zeros(batch_size, 1, requires_grad=True, device=self.device).long()
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat([start_token, target[:, :-1]], dim=1)
        h = self.seq.init_hidden(batch_size)

        log_probs = torch.zeros(batch_size, requires_grad=True, device=self.device)
        entropy = torch.zeros(batch_size, requires_grad=True, device=self.device)

        for step in range(seq_length):
            logits, h = self.seq(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            log_probs += self.NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob*prob), dim=1)

        return log_probs, entropy


    def NLLLoss(self, inputs, targets):
        target_expanded = torch.zeros(inputs.size()).to(self.device)

        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)  # one-hot encode the indices
        loss = torch.tensor(target_expanded, requires_grad=True, device=self.device) * inputs
        loss = torch.sum(loss, 1)
        return loss


    def sample(self, batch_size, max_length=140):
        start_token = torch.zeros(batch_size, requires_grad=True, device=self.device).long()
        start_token[:] = self.voc.vocab['GO']
        h = self.seq.init_hidden(batch_size)
        x = start_token

        sequences = []
        log_probs = torch.zeros(batch_size, requires_grad=True, device=self.device)
        finised = torch.zeros(batch_size).byte().to(self.device)
        entropy = torch.zeros(batch_size, requires_grad=True, device=self.device)

        for _ in range(max_length):
            logits, h = self.seq(x, h)
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            x = torch.multinomial(prob, 1).view(-1)
            
            sequences.append(x.view(-1, 1))
            log_probs += self.NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), dim=1)

            x = torch.tensor(x.data, requires_grad=True, device=self.device)

            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
        
        sequences = torch.cat(sequences, 1)

        return sequences.data, log_probs, entropy

    



