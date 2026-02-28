"""Run a minimal HGT forward pass."""

import torch

from graph_transformer.hgt import ScratchHGTModel


def main() -> None:
    torch.manual_seed(42)
    num_nodes = 6
    num_types = 2
    num_relations = 2
    in_dim = 16
    hidden_dim = 16
    out_dim = 4

    node_feat = torch.randn(num_nodes, in_dim)
    node_type = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    model = ScratchHGTModel(num_types=num_types, num_relations=num_relations, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    model.eval()
    with torch.no_grad():
        output = model(node_feat, node_type, edge_index, edge_type)
    print(f"HGT output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
