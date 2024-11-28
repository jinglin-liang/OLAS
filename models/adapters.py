import torch.nn as nn
from axial_attention import AxialImageTransformer


class AxialTransformerAdapter(nn.Module):
    def  __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_size=768,
        num_layers=5,
        heads=8,
        reversible=True
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_size, 1)
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_size, out_channels, 1)
        )
        self.transformer = AxialImageTransformer(
            dim = hidden_size,
            depth = num_layers,
            heads = heads,
            reversible = reversible
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.transformer(x)
        x = self.output_conv(x)
        return x
