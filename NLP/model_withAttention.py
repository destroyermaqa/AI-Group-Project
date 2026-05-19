import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # batch_first=True matches your preprocessing pipeline
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        
        # We must return encoder_outputs to compute attention over all source time steps
        encoder_outputs, hidden = self.rnn(embedded)
        # encoder_outputs shape: [batch_size, src_len, hidden_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]
        return encoder_outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Linear layers for the additive equation: v_a^T * tanh(W_a * s_t + U_a * h_i)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder hidden state (s_t)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder outputs (h_i)
        self.v = nn.Linear(hidden_dim, 1, bias=False)           # For alignment scores (v_a)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: [num_layers, batch_size, hidden_dim] -> we take the last layer
        # encoder_outputs shape: [batch_size, src_len, hidden_dim]
        
        s_t = decoder_hidden[-1].unsqueeze(1) # shape: [batch_size, 1, hidden_dim]
        
        # Broadcast addition computes score for every src_len token step
        # energy shape: [batch_size, src_len, hidden_dim]
        energy = torch.tanh(self.W(s_t) + self.U(encoder_outputs))
        
        # scores shape: [batch_size, src_len]
        scores = self.v(energy).squeeze(2)
        
        # Normalize weights across the source sentence sequence dimension
        return F.softmax(scores, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.attention = BahdanauAttention(hidden_dim)
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # GRU input size is (embedding dimension + context vector dimension)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        
        # Fully connected layer combines GRU hidden state, context, and raw token embedding
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, encoder_outputs):
        # input_token shape: [batch_size]
        # hidden shape: [num_layers, batch_size, hidden_dim]
        # encoder_outputs shape: [batch_size, src_len, hidden_dim]
        
        input_token = input_token.unsqueeze(1) # shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token)) # shape: [batch_size, 1, emb_dim]
        
        # 1. Get alignment weights: [batch_size, 1, src_len]
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        
        # 2. Batch matrix multiplication to get weighted context vector
        # [batch_size, 1, src_len] x [batch_size, src_len, hidden_dim] -> [batch_size, 1, hidden_dim]
        context = torch.bmm(a, encoder_outputs)
        
        # 3. Concatenate embedding and context vector as input to GRU
        rnn_input = torch.cat((embedded, context), dim=2) # shape: [batch_size, 1, emb_dim + hidden_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 4. Remove time step dimension to pass through fully connected output mapping
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1)) # shape: [batch_size, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [batch_size, src_len]
        # trg shape: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Run source sequence entirely through encoder
        encoder_outputs, hidden = self.encoder(src)
        
        # Seed decoder with initial <SOS> token input
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = prediction
            
            top1 = prediction.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs
