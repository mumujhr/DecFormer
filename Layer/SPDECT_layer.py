import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, calibration_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
            
        attn = F.softmax(attn, dim=-1)

        if calibration_mask is not None:
            attn = attn * calibration_mask
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, calibration_mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if calibration_mask is not None:
            calibration_mask = calibration_mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, calibration_mask=calibration_mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn


class FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class SPDECTLayer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(SPDECTLayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)
        self.attn = None

    def forward(self, x, calibration_mask=None, need_attn=False):
        x = self.node_norm(x)
        x = x.unsqueeze(0)
        x, _ = self.node_transformer(x, x, x, calibration_mask=calibration_mask)
        x = self.node_ffn(x)
        x = x.squeeze(0)

        return x
