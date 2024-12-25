import torch
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
        # input x to axial transformer
        x = self.input_conv(x)
        x = self.transformer(x)
        x = self.output_conv(x)
        # from diagonal to sequence
        idx_tensor = torch.arange(x.size(-1)).to(x.device)
        x = x[:, :, idx_tensor, idx_tensor]
        x = x.transpose(1, 2).contiguous()
        return x


class AxialTransformerRnnAdapter(nn.Module):
    def  __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_size=768,
        axial_tf_layers=5,
        rnn_layers=2,
        heads=8,
        reversible=True
    ):
        super().__init__()
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads."
        assert hidden_size % 2 == 0, "hidden_size must be divisible by 2."
        self.input_conv = nn.Conv2d(in_channels, hidden_size, 1)
        self.transformer = AxialImageTransformer(
            dim = hidden_size,
            depth = axial_tf_layers,
            heads = heads,
            reversible = reversible
        )
        self.rnn = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.LSTM(
                hidden_size, 
                hidden_size // 2, 
                num_layers=rnn_layers, 
                batch_first=True, 
                bidirectional=True
            )
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.output_fc = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, out_channels)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # input x to axial transformer
        x = self.input_conv(x)
        x = self.transformer(x)
        # from diagonal to sequence
        idx_tensor = torch.arange(x.size(-1)).to(x.device)
        x = x[:, :, idx_tensor, idx_tensor]
        x = x.transpose(1, 2).contiguous()
        # input sequence to rnn
        x_rnn, _ = self.rnn(self.dropout(x))
        x_rnn = self.ln(x_rnn)
        # residual connection
        x = x + x_rnn
        # output x
        x = self.output_fc(self.dropout(x))
        return x
