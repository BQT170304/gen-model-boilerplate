import torch
from torch import Tensor
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    ### Dense Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 drop_rate: float = 0.,
                 d_t_emb: int = None) -> None:
        """
        in_channels: the number of input channels
        out_channels: is the number of out channels. defaults to `in_channels.
        drop_rate: parameter of dropout layer
        d_t_emb: the size of timestep embeddings if not None. defaults to None
        """
        super().__init__()

        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        out_channels //= 2

        if d_t_emb is not None:
            # Time step embeddings
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=d_t_emb, out_features=out_channels),
            )

        # Normalization and convolution in input layer
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        # Normalization and convolution in output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """

        # Normalization and convolution in input layer
        out = self.in_layers(x)

        if t_emb is not None:
            # Time step embeddings
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            # Add time step embeddings
            out = out + t_emb[:, :, None, None]

        # Normalization and convolution in output layers
        out = self.out_layers(x)

        # concat in dense_block
        return torch.cat((out, x), dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 32, 10, 10)
    t = torch.randn(2, 32)
    denseBlock_timeEmbedding = DenseBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = denseBlock_timeEmbedding(x, t)
    print('***** DenseBlock_with_TimeEmbedding *****')
    print('Input:', x.shape, t.shape)
    print('Output:', out1.shape)

    print('-' * 60)

    denseBlock = DenseBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = denseBlock(x)
    print('***** DenseBlock_without_TimeEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out2.shape)