from collections import OrderedDict

import torch
import torch.nn as nn
from axial_attention import AxialImageTransformer


class AxialTransformerAdapter(nn.Module):
    def  __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_size=768,
        axial_tf_layers=5,
        heads=8,
        reversible=True,
        **kwargs
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_size, 1)
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_size, out_channels, 1)
        )
        self.transformer = AxialImageTransformer(
            dim = hidden_size,
            depth = axial_tf_layers,
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
        reversible=True,
        **kwargs
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
    

# class AxialTransformerTfAdapter(nn.Module):
#     def  __init__(
#         self,
#         in_channels=3,
#         out_channels=3,
#         hidden_size=768,
#         axial_tf_layers=5,
#         seq_layers=2,
#         heads=8,
#         reversible=True
#     ):
#         super().__init__()
#         assert hidden_size % heads == 0, "hidden_size must be divisible by heads."
#         assert hidden_size % 2 == 0, "hidden_size must be divisible by 2."
#         self.input_conv = nn.Conv2d(in_channels, hidden_size, 1)
#         self.transformer = AxialImageTransformer(
#             dim = hidden_size,
#             depth = axial_tf_layers,
#             heads = heads,
#             reversible = reversible
#         )
#         self.seq_model = nn.Sequential(
#             nn.LeakyReLU(inplace=True),
#             nn.Transformer(
#                 hidden_size, 
#                 hidden_size // 2, 
#                 num_layers=seq_layers, 
#                 batch_first=True, 
#                 bidirectional=True
#             )
#         )
#         self.ln = nn.LayerNorm(hidden_size)
#         self.output_fc = nn.Sequential(
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(hidden_size, out_channels)
#         )
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         # input x to axial transformer
#         x = self.input_conv(x)
#         x = self.transformer(x)
#         # from diagonal to sequence
#         idx_tensor = torch.arange(x.size(-1)).to(x.device)
#         x = x[:, :, idx_tensor, idx_tensor]
#         x = x.transpose(1, 2).contiguous()
#         # input sequence to rnn
#         x_rnn, _ = self.rnn(self.dropout(x))
#         x_rnn = self.ln(x_rnn)
#         # residual connection
#         x = x + x_rnn
#         # output x
#         x = self.output_fc(self.dropout(x))
#         return x


# from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, unet_init_features=32, **kwargs):
        super(UNet, self).__init__()

        features = unet_init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.conv(dec1)
        # from diagonal to sequence
        idx_tensor = torch.arange(output.size(-1)).to(output.device)
        output = output[:, :, idx_tensor, idx_tensor]
        output = output.transpose(1, 2).contiguous()
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
