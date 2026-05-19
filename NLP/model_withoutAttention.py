import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden shape: [num_layers, batch_size, hidden_dim]
        return hidden

class VanillaDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden):
        # input_token shape: [batch_size] -> processed step-by-step
        input_token = input_token.unsqueeze(1) # shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1)) # shape: [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder passes single context vector (hidden)
        hidden = self.encoder(src)
        
        # Initial target input token is <SOS>
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = prediction
            top1 = prediction.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs
