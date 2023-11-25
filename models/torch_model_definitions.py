import torch
import torch.nn as nn

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
        h_0 = (torch.zeros(self.num_layers * self.h_n_dim, batch_size, self.hidden_size)
               .requires_grad_().to(MODEL_DEFINITION_DEVICE))
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


class BahdanauAttention(nn.Module):
    """https://github.com/sooftware/attentions/blob/master/attentions.py"""
    def __init__(self, hidden_dim: int) -> None:
        super(BahdanauAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = nn.functional.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn


class AttentionGRUDecoder(nn.Module):
    def __init__(self, embedded_size, bidirectional=False, dropout=0.0, noise=0.0):
        super(AttentionGRUDecoder, self).__init__()
        self.h_n_dim = 2 if bidirectional else 1
        self.gru = nn.GRU(embedded_size * self.h_n_dim, embedded_size, 1,
                          bidirectional=bidirectional, batch_first=True)
        self.attention = BahdanauAttention(embedded_size * self.h_n_dim)
        self.fc = nn.Sequential(
            nn.Flatten(1, -1),
            GaussianNoise(noise),
            nn.Dropout(dropout),
            nn.Linear(embedded_size * self.h_n_dim, 1),
        )

    def forward(self, y_prev, hidden, enc_out):
        query = hidden.permute(1, 0, 2).reshape(-1, 1, self.h_n_dim * self.gru.hidden_size)
        attn_out, attn_weights = self.attention(query, enc_out, enc_out)

        out, hidden = self.gru(attn_out + y_prev, hidden)
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
        self.adec = AttentionGRUDecoder(embedding_size, bidirectional=bidirectional,
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


# endregion


# region Positional encoding


"""https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6"""

# endregion
