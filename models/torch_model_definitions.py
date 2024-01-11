import torch
import torch.nn as nn
from math import log as math_log
from math import sqrt as math_sqrt


MODEL_DEFINITION_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    MODEL_DEFINITION_DEVICE = torch.device("cuda")

# region GaussianNoise, from: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            noise = self.noise.expand(*x.size()).float().normal_() * scale
            return x + noise
        return x


# endregion


# region Encoder-decoder


class GRUEncoder(nn.Module):
    def __init__(self, features, embedding_size, num_layers=1, bidirectional=False, dropout=0.0, noise=0.0):
        super(GRUEncoder, self).__init__()
        self.hidden_size = embedding_size
        self.num_layers = num_layers
        self.h_n_dim = 2 if bidirectional else 1
        self.noise = GaussianNoise(0.0)
        self.gru = nn.GRU(features, embedding_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.noise(x)
        h_0 = (torch.zeros(self.num_layers * self.h_n_dim, batch_size, self.hidden_size).to(MODEL_DEFINITION_DEVICE))
        out, hidden = self.gru(x, h_0)

        return out, hidden


class GRUDecoder(nn.Module):
    def __init__(self, features, embedded_size, num_layers=1, bidirectional=False, dropout=0.0, noise=0.0):
        super(GRUDecoder, self).__init__()
        self.h_n_dim = 2 if bidirectional else 1
        self.gru = nn.GRU(features, embedded_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.flatten = nn.Flatten(1, -1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedded_size * num_layers * self.h_n_dim, 1)
        self.noise = GaussianNoise(noise)

    def forward(self, x, h):
        x, hidden = self.gru(x, h)

        out = hidden.permute(1, 0, 2)
        out = self.flatten(out)
        out = self.noise(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out,hidden


class Seq2seq(nn.Module):
    def __init__(self, features=11, pred_len=3, embedding_size=64, num_layers=1, bidirectional=False,
                 dropout=0.2, in_noise=0.0, out_noise=0.0, **kwargs):
        super(Seq2seq, self).__init__()
        self.pred_len = pred_len
        self.features = features
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.enc = GRUEncoder(features, embedding_size, num_layers, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.0, noise=in_noise)
        self.dec = GRUDecoder(1, embedding_size, num_layers, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.0, noise=out_noise)

    def forward(self, x, y=None, teacher_forcing=0.0):
        batch_size = x.shape[0]
        _, hidden = self.enc(x)
        dec_input = x[:, -1, 0].reshape(-1, 1, 1)  # this will be y_prev in my case
        output = torch.zeros(batch_size, self.pred_len).to(MODEL_DEFINITION_DEVICE)

        for i in range(self.pred_len):
            out, hidden = self.dec(dec_input, hidden)
            output[:, i] = out[:, 0]
            if y is not None and torch.rand(1) < teacher_forcing:
                dec_input = y[:, i].reshape(-1, 1, 1)
            else:
                dec_input = out.unsqueeze(1)

        return output


# endregion


# region Attention encoder-decoder


class Attention(nn.Module):
    """from: https://github.com/sooftware/attentions/blob/master/attentions.py"""

    def __init__(self, dim: int):
        super(Attention, self).__init__()
        self.sqrt_dim = math_sqrt(dim)

    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class AttentionGRUDecoder(nn.Module):
    def __init__(self, input_size, embedded_size, bidirectional=False, dropout=0.0, noise=0.0):
        super(AttentionGRUDecoder, self).__init__()
        self.h_n_dim = 2 if bidirectional else 1
        self.gru = nn.GRU(embedded_size * self.h_n_dim + input_size, embedded_size, 1,
                          bidirectional=bidirectional, batch_first=True)
        self.attention = Attention(embedded_size * self.h_n_dim)
        self.fc = nn.Sequential(
            nn.Flatten(1, -1),
            GaussianNoise(noise),
            nn.Dropout(dropout),
            nn.Linear(embedded_size * self.h_n_dim, 1),
        )

    def forward(self, y_prev, hidden, enc_out):
        query = hidden.permute(1, 0, 2).reshape(-1, 1, self.h_n_dim * self.gru.hidden_size)
        attn_out, attn_weights = self.attention(query, enc_out, enc_out)

        inp = torch.cat((y_prev, attn_out), dim=2)
        out, hidden = self.gru(inp, hidden)

        out = self.fc(out)

        return out, hidden


class AttentionSeq2seq(nn.Module):
    def __init__(self, features=11, pred_len=3, embedding_size=10, bidirectional=False,
                 dropout=0.2, in_noise=0.0, out_noise=0.0, **kwargs):
        super(AttentionSeq2seq, self).__init__()
        self.pred_len = pred_len
        self.features = features
        self.embedding_size = embedding_size
        self.enc = GRUEncoder(features, embedding_size, 1, bidirectional=bidirectional,
                              dropout=0.0, noise=in_noise)
        self.adec = AttentionGRUDecoder(1, embedding_size, bidirectional=bidirectional,
                                        dropout=dropout, noise=out_noise)

    def forward(self, x, y=None, teacher_forcing=0.0):
        batch_size = x.shape[0]
        enc_out, hidden = self.enc(x)
        dec_input = x[:, -1, 0].reshape(-1, 1, 1)  # this will be y_prev in my case
        output = torch.zeros(batch_size, self.pred_len).to(MODEL_DEFINITION_DEVICE)

        for i in range(self.pred_len):
            out, hidden = self.adec(dec_input, hidden, enc_out)

            output[:, i] = out[:, 0]

            if y is not None and torch.rand(1) < teacher_forcing:
                dec_input = y[:, i].reshape(-1, 1, 1)
            else:
                dec_input = out.unsqueeze(1)

        return output


# print(AttentionSeq2seq(bidirectional=True).to(MODEL_DEFINITION_DEVICE)(torch.zeros(1, 3, 11).to(MODEL_DEFINITION_DEVICE)))
# endregion


# region Positional encoding


class ConcPositionalEncodingS2S(nn.Module):
    """Positional encoding via concatenation"""
    def __init__(self, d_model, max_seq):
        super(ConcPositionalEncodingS2S, self).__init__()

        pe = torch.zeros(max_seq, d_model)
        position = torch.arange(0, max_seq, 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * -(math_log(10_000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).to(MODEL_DEFINITION_DEVICE)

    def forward(self, x):
        expanded = self.pe[:, :x.shape[1]].expand((x.shape[0], -1, -1))
        return torch.cat((x, expanded), dim=2)

    def code_index(self, x, i):
        expanded = self.pe[:, i].expand((x.shape[0], -1, -1))
        return torch.cat((x, expanded), dim=2)


class PosAttSeq2seq(nn.Module):
    def __init__(self, features=11, pred_len=3, embedding_size=10, bidirectional=False,
                 dropout=0.2, in_noise=0.0, out_noise=0.0, max_seq=72, **kwargs):
        super(PosAttSeq2seq, self).__init__()
        self.pred_len = pred_len
        self.features = features
        self.embedding_size = embedding_size
        self.pe = ConcPositionalEncodingS2S(1, max_seq)
        self.enc = GRUEncoder(features + 1, embedding_size, 1, bidirectional=bidirectional,
                              dropout=0.0, noise=in_noise)
        self.adec = AttentionGRUDecoder(2, embedding_size, bidirectional=bidirectional,
                                        dropout=dropout, noise=out_noise)

    def forward(self, x, y=None, teacher_forcing=0.0):
        batch_size = x.shape[0]
        enc_out, hidden = self.enc(self.pe(x))
        dec_input = x[:, -1, 0].reshape(-1, 1, 1)  # this will be y_prev in my case
        output = torch.zeros(batch_size, self.pred_len).to(MODEL_DEFINITION_DEVICE)

        for i in range(self.pred_len):
            dec_input = self.pe.code_index(dec_input, i)  # pos encode decoder out
            out, hidden = self.adec(dec_input, hidden, enc_out)

            output[:, i] = out[:, 0]

            if y is not None and torch.rand(1) < teacher_forcing:
                dec_input = y[:, i].reshape(-1, 1, 1)
            else:
                dec_input = out.unsqueeze(1)

        return output


class AddPositionalEncodingS2S(nn.Module):
    """Positional encoding via addition"""
    def __init__(self, d_model, max_seq):
        super(AddPositionalEncodingS2S, self).__init__()

        pe = torch.zeros(max_seq, d_model)
        position = torch.arange(0, max_seq, 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * -(math_log(10_000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).to(MODEL_DEFINITION_DEVICE)

    def forward(self, x):
        expanded = self.pe[:, :x.shape[1]].expand((x.shape[0], -1, -1))
        return x + expanded

    def code_index(self, x, i):
        expanded = self.pe[:, i].expand((x.shape[0], -1, -1))
        return x + expanded


class AddPosAttSeq2seq(nn.Module):
    def __init__(self, features=11, pred_len=3, embedding_size=10, bidirectional=False,
                 dropout=0.2, in_noise=0.0, out_noise=0.0, max_seq=72, **kwargs):
        super(AddPosAttSeq2seq, self).__init__()
        self.pred_len = pred_len
        self.features = features
        self.embedding_size = embedding_size
        self.pe = AddPositionalEncodingS2S(1, max_seq)
        self.enc = GRUEncoder(features, embedding_size, 1, bidirectional=bidirectional,
                              dropout=0.0, noise=in_noise)
        self.adec = AttentionGRUDecoder(1, embedding_size, bidirectional=bidirectional,
                                        dropout=dropout, noise=out_noise)

    def forward(self, x, y=None, teacher_forcing=0.0):
        batch_size = x.shape[0]
        enc_out, hidden = self.enc(x)
        enc_out = self.pe(enc_out)
        dec_input = x[:, -1, 0].reshape(-1, 1, 1)  # this will be y_prev in my case
        output = torch.zeros(batch_size, self.pred_len).to(MODEL_DEFINITION_DEVICE)

        for i in range(self.pred_len):
            hidden = self.pe.code_index(hidden, i)
            out, hidden = self.adec(dec_input, hidden, enc_out)

            output[:, i] = out[:, 0]

            if y is not None and torch.rand(1) < teacher_forcing:
                dec_input = y[:, i].reshape(-1, 1, 1)
            else:
                dec_input = out.unsqueeze(1)

        return output


# print(ConcPositionalEncodingS2S(1, 72).code_index((torch.zeros(2, 1, 1).to(MODEL_DEFINITION_DEVICE)), 2))
# print(PosAttSeq2seq(bidirectional=True).to(MODEL_DEFINITION_DEVICE)(torch.zeros(1, 3, 11).to(MODEL_DEFINITION_DEVICE)))

# print(AddPositionalEncodingS2S(1, 72)(torch.zeros(2, 3, 20).to(MODEL_DEFINITION_DEVICE)))
# print(AddPosAttSeq2seq(bidirectional=True).to(MODEL_DEFINITION_DEVICE)(torch.zeros(1, 3, 11).to(MODEL_DEFINITION_DEVICE)))
# endregion

