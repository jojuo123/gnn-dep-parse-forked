import torch
from torch import nn 

class GATNetwork(nn.Module):
    def __init__(self, 
                 in_features: int,
                 n_model: int,
                 n_layers: int = 1,
                 n_heads: int = 8,
                 is_concat: bool = True,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False,
                 selfloop: bool = True,
                 dropout: float = 0.,
                 layer_dropout: float = 0.6,
                 activation: bool = True,
                 norm: bool = False,
                 layer: str = 'v1',
                 residual: bool = True,
                ):
        super().__init__()

        assert layer == 'v1' or layer == 'v2'

        self.in_features = in_features
        self.n_model = n_model
        self.n_layers = n_layers
        self.selfloop = selfloop
        self.norm = norm
        self.layer = layer
        self.activation = activation
        self.residual = residual

        self.gat_layers = nn.ModuleList([
            nn.Sequential(
                GraphAttentionLayer(n_model if i > 0 else in_features, n_model, n_heads, is_concat, layer_dropout, leaky_relu_negative_slope) if layer == 'v1' else GraphAttentionV2Layer(n_model if i > 0 else in_features, n_model, n_heads, is_concat, layer_dropout, leaky_relu_negative_slope, share_weights),
                nn.ELU() if (activation and i < (n_layers-1) and n_layers > 1) else nn.Identity(),
                nn.LayerNorm([n_model]) if norm else nn.Identity()
            )
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def __repr__(self):
        s = f"n_model={self.n_model}, n_layers={self.n_layers}, layer={self.layer}"
        if self.activation:
            s += f', activation={self.activation}'
        if self.selfloop:
            s += f", selfloop={self.selfloop}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        if self.norm:
            s += f", norm={self.norm}"
        if self.residual:
            s += f', residual={self.residual}'
        return f"{self.__class__.__name__}({s})"
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask = None) -> torch.Tensor:
        if self.selfloop:
            adj.diagonal(0, 1, 2).fill_(1.)
        if mask is not None:
            adj = adj.masked_fill(~mask.unsqueeze(1 & mask.unsqueeze(2)), 0)
        for gat, activation, norm in self.gat_layers:
            out = self.dropout(activation(gat(x, adj)))
            x = out if not self.residual else x + out
            x = norm(x)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 n_heads: int = 1,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2
                ):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.SoftMax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat, g_repeat_interleave], dim=1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)
        e = e.masked_fill(adj == 0, float('-inf'))

        assert adj.shape[0] == 1 or adj.shape[0] == n_nodes
        assert adj.shape[1] == 1 or adj.shape[1] == n_nodes
        assert adj.shape[2] == 1 or adj.shape[2] == self.n_heads

        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden) if self.is_concat else attn_res.mean(dim=1)
    
    def __repr__(self):
        s = f"n_model={self.in_features}, is_concat={self.is_concat}, n_heads={self.n_heads}, n_hidden={self.n_hidden}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"
        
class GraphAttentionV2Layer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False
                ):
        super(GraphAttentionV2Layer, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads 
        else:
            self.n_hidden = out_features
        
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        n_nodes = h.shape[0]
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        e = e.masked_fill(adj == 0, float('-inf'))

        assert adj.shape[0] == 1 or adj.shape[0] == n_nodes
        assert adj.shape[1] == 1 or adj.shape[1] == n_nodes
        assert adj.shape[2] == 1 or adj.shape[2] == self.n_heads

        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden) if self.is_concat else attn_res.mean(dim=1)
    
    def __repr__(self):
        s = f"n_model={self.in_features}, is_concat={self.is_concat}, n_heads={self.n_heads}, n_hidden={self.n_hidden}"
        if self.share_weights:
            s += f', share_weights={self.share_weights}'
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"