import torch.nn as nn
import torch
import torch.nn.functional as F

class UNetDownBlock(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      pad_maxpool=0,
      normgroups=4
    ):
    super(UNetDownBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size=3,
      padding=1
    )
    self.groupnorm1 = nn.GroupNorm(
      normgroups,
      out_channels
    )
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(
      out_channels,
      out_channels,
      kernel_size=3,
      padding=1
    )
    self.groupnorm2 = nn.GroupNorm(
      normgroups,
      out_channels
    )
    self.relu2 = nn.ReLU()

    self.conv3 = nn.Conv2d(
      out_channels,
      out_channels,
      kernel_size=3,
      padding=1
    )
    self.groupnorm3 = nn.GroupNorm(
      normgroups,
      out_channels
    )
    self.relu3 = nn.ReLU()
    self.maxpool = nn.MaxPool2d(
      2, padding=pad_maxpool
    )

  def forward(self, x):
    # First convolution
    x = self.conv1(x)
    x = self.groupnorm1(x)
    x_for_skip = self.relu1(x)

    # Second convolution
    x = self.conv2(x_for_skip)
    x = self.groupnorm2(x)
    x = self.relu2(x)

    x = self.conv3(x)
    x = self.groupnorm3(x)

    # Skip connection
    x = x + x_for_skip
    x = self.relu3(x)

    x = self.maxpool(x)
    return x


class UNetUpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_dim, normgroups=4):
    super(UNetUpBlock, self).__init__()
    self.upsample =  nn.Upsample(scale_factor=2, mode='nearest')

    # Convolution 1
    self.conv1 = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1
    )
    self.groupnorm1 = nn.GroupNorm(normgroups, out_channels)
    self.relu1 = nn.ReLU()

    # Convolution 2
    self.conv2 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, padding=1
    )
    self.groupnorm2 = nn.GroupNorm(normgroups, out_channels)
    self.relu2 = nn.ReLU()

    # Convolution 3
    self.conv3 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, padding=1
    )
    self.groupnorm3 = nn.GroupNorm(normgroups, out_channels)
    self.relu3 = nn.ReLU()

    # Parameters to scale and shift the time embedding
    self.time_mlp = nn.Linear(time_dim, time_dim)
    self.time_relu = nn.ReLU()


  def forward(self, x, x_down, t_embed):
    x_up = self.upsample(x)
    #print("x_up: ", x_up.shape)
    x = torch.cat([x_down, x_up], dim=1)

    # Cut embedding to be the size of the current channels
    t_embed = t_embed[:,:x.shape[1]]

    # Enable the neural network to modify the time-embedding
    # as it needs to
    t_embed = self.time_mlp(t_embed)
    t_embed = self.time_relu(t_embed)
    t_embed = t_embed[:,:,None,None].expand(x.shape)

    # Add time-embedding to input.
    x = x + t_embed

    # Convolution 1
    x = self.conv1(x)
    x = self.groupnorm1(x)
    x_for_skip = self.relu1(x)

    # Convolution 2
    x = self.conv2(x_for_skip)
    x = self.groupnorm2(x)
    x = self.relu2(x)

    # Convolution 3
    x = self.conv3(x)
    x = self.groupnorm3(x)

    # Skip connection
    x = x + x_for_skip
    x = self.relu3(x)

    return x


class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    # Down blocks
    self.down1 = UNetDownBlock(1, 4, normgroups=1)
    self.down2 = UNetDownBlock(4, 10, normgroups=1)
    self.down3 = UNetDownBlock(10, 20, normgroups=2)
    self.down4 = UNetDownBlock(20, 40, normgroups=4)

    # Convolutional layer at the bottom of the U-Net
    self.bottom_conv = nn.Conv2d(
        40, 40, kernel_size=3, padding=1
    )
    self.bottom_groupnorm = nn.GroupNorm(4, 40)
    self.bottom_relu = nn.ReLU()

    # Up blocks
    self.up1 = UNetUpBlock(60, 20, 60, normgroups=2) # down4 channels + down3 channels
    self.up2 = UNetUpBlock(30, 10, 30, normgroups=1) # down2 channels + up1 channels
    self.up3 = UNetUpBlock(14, 5, 14, normgroups=1) # down1 channels + up2 channels
    self.up4 = UNetUpBlock(6, 5, 6, normgroups=1) # input channels + up3 channels

    # Final convolution to produce output. This layer injects negative
    # values into the output.
    self.final_conv = nn.Conv2d(
        5, 1, kernel_size=3, padding=1
    )

  def forward(self, x, t_emb):
    """
    Parameters
    ----------
      x: Input tensor representing an image
      t_embed: The time-embedding vector for the current timestep
    """
    # Pad the input so that it is 32x32. This enables downsampling to
    # 16x16, then to 8x8, and finally to 4x4 at the bottom of the "U"
    x = F.pad(x, (2,2,2,2), 'constant', 0)

    # Down-blocks of the U-Net compress the image down to a smaller
    # representation
    x_d1 = self.down1(x)
    x_d2 = self.down2(x_d1)
    x_d3 = self.down3(x_d2)
    x_d4 = self.down4(x_d3)

    # Bottom layer perform final transformation on compressed representation
    # before re-inflation
    x_bottom = self.bottom_conv(x_d4)
    x_bottom = self.bottom_groupnorm(x_d4)
    x_bottom = self.bottom_relu(x_d4)

    # Up-blocks re-inflate the compressed representation back to the original
    # image size while taking as input various representations produced in the
    # down-sampling steps
    x_u1 = self.up1(x_bottom, x_d3, t_emb)
    x_u2 = self.up2(x_u1, x_d2, t_emb)
    x_u3 = self.up3(x_u2, x_d1, t_emb)
    x_u4 = self.up4(x_u3, x, t_emb)

    # Final convolutional layer. Introduces negative values.
    x_u4 = self.final_conv(x_u4)

    # Remove initial pads to produce a 28x28 MNIST digit
    x_u4 = x_u4[:,:,2:-2,2:-2]

    return x_u4