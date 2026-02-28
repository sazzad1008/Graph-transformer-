"""Run a minimal Vision Transformer forward pass."""

import torch

from graph_transformer.vit import ViT


def main() -> None:
    torch.manual_seed(42)
    model = ViT(in_channels=3, patch_size=16, emb_size=256, img_size=224, depth=4, n_classes=10)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"ViT output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
