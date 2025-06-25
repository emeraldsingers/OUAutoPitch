import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import config
import math

class Encoder(nn.Module):
    def __init__(self, phoneme_vocab_size, phoneme_embedding_dim, note_feature_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embedding_dim)
        lstm_input_dim = phoneme_embedding_dim + note_feature_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout, 
            batch_first=True,
            bidirectional=True
        )
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(lstm_input_dim)

    def forward(self, note_features, phonemes):
        embedded_phonemes = self.dropout(self.phoneme_embedding(phonemes))
        
        lstm_input = torch.cat((embedded_phonemes, note_features), dim=2)
        lstm_input = self.pos_encoding(lstm_input)
        outputs, (hidden, cell) = self.lstm(lstm_input)
        
        hidden = torch.cat((hidden[0:self.num_layers,:,:], hidden[self.num_layers:self.num_layers*2,:,:]), dim=2)
        cell = torch.cat((cell[0:self.num_layers,:,:], cell[self.num_layers:self.num_layers*2,:,:]), dim=2)

        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        
        return outputs, hidden, cell

class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, n_heads):
        super().__init__()
        
        assert decoder_hidden_dim % n_heads == 0
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.n_heads = n_heads
        self.head_dim = decoder_hidden_dim // n_heads
        
        self.fc_q = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.fc_k = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.fc_v = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        
        self.fc_o = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(config.DEVICE)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.unsqueeze(1).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.decoder_hidden_dim)
        
        x = self.fc_o(x)
        
        return x, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = attention
        
        decoder_input_dim = 3 # pitch points
        
        self.lstm = nn.LSTM(
            hidden_dim + decoder_input_dim,
            hidden_dim, 
            num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + decoder_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_input, hidden, cell, encoder_outputs):
        context_vector, _ = self.attention(hidden[-1], encoder_outputs, encoder_outputs)
        
        lstm_input = torch.cat((context_vector, decoder_input), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        prediction_input = torch.cat((decoder_input.squeeze(1), output.squeeze(1), context_vector.squeeze(1)), dim=1)
        
        prediction = self.fc_out(prediction_input)
        
        return prediction.unsqueeze(1), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_features, encoder_phonemes, decoder_input, teacher_forcing_ratio=0.5):
        batch_size = decoder_input.shape[0]
        target_len = decoder_input.shape[1]
        decoder_output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, target_len, decoder_output_dim).to(config.DEVICE)
        
        encoder_outputs, hidden, cell = self.encoder(encoder_features, encoder_phonemes)
        
        current_input_point = decoder_input[:, 0, :].unsqueeze(1)

        for t in range(target_len):

            output, hidden, cell = self.decoder(current_input_point, hidden, cell, encoder_outputs)
            
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            if teacher_force and t < target_len - 1:
                current_input_point = decoder_input[:, t+1, :].unsqueeze(1)
            else:
                current_input_point = output
            
        return outputs 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe shape: [max_len, 1, d_model]
        
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)  # [batch_size, seq_len, d_model]