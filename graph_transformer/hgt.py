"""Heterogeneous graph transformer layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class ScratchHGTConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_types: int, num_relations: int, n_heads: int = 4):
        super().__init__()
        self.out_dim, self.n_heads = out_dim, n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = self.d_k**0.5

        self.k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.a_linears = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_types)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_types)])

        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(
        self, node_feat: torch.Tensor, node_type: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = node_feat.size(0)

        k_all = torch.zeros(num_nodes, self.out_dim, device=node_feat.device)
        q_all = torch.zeros(num_nodes, self.out_dim, device=node_feat.device)
        v_all = torch.zeros(num_nodes, self.out_dim, device=node_feat.device)

        for t in range(len(self.k_linears)):
            mask = node_type == t
            if mask.any():
                k_all[mask] = self.k_linears[t](node_feat[mask])
                q_all[mask] = self.q_linears[t](node_feat[mask])
                v_all[mask] = self.v_linears[t](node_feat[mask])

        k, q, v = [x.view(-1, self.n_heads, self.d_k) for x in (k_all, q_all, v_all)]

        # k_rel: (edges, heads, d_k) relation-specific key transform
        k_rel = torch.einsum("ehd,ehdf->ehf", k[src], self.relation_att[edge_type])
        att_score = (q[dst] * k_rel).sum(dim=-1) / self.sqrt_dk
        att_weight = softmax(att_score, dst, num_nodes=num_nodes)

        # v_rel: (edges, heads, d_k) relation-specific value transform
        v_rel = torch.einsum("ehd,ehdf->ehf", v[src], self.relation_msg[edge_type])
        messages = (v_rel * att_weight.unsqueeze(-1)).reshape(-1, self.out_dim)

        aggr_out = scatter_add(messages, dst, dim=0, dim_size=num_nodes)

        final_out = aggr_out.clone()
        for t in range(len(self.a_linears)):
            mask = node_type == t
            if mask.any():
                final_out[mask] = self.norms[t](self.a_linears[t](aggr_out[mask]))
        return final_out


class ScratchHGTModel(nn.Module):
    def __init__(self, num_types: int, num_relations: int, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = ScratchHGTConv(hidden_dim, hidden_dim, num_types, num_relations)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, node_type: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        h = F.gelu(self.input_proj(x))
        h = self.conv1(h, node_type, edge_index, edge_type)
        return self.classifier(h)
