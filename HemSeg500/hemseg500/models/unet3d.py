from monai.networks.nets import UNet

def unet():
        unet = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(32, 64, 128, 256,512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
        )
        return unet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer, activation, conv_bias=True,dropout_rate = 0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=conv_bias)
        self.norm = norm_layer(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class UNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_rate = 0.5):
        super(UNetBottleneck, self).__init__()
        
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.batch_norm1 = nn.InstanceNorm3d(out_channels)
        self.batch_norm2 = nn.InstanceNorm3d(out_channels)
        
        self.dropout1 = nn.Dropout3d(dropout_rate)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x):
            
        x = self.downsample(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.upsample(x)


        return x

class CustomUNet3D(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(CustomUNet3D, self).__init__()
        
        self.encoder1 = nn.Sequential(
            ConvBlock(1, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(32, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
        )
        
        self.encoder2 = nn.Sequential(
            ConvBlock(32, 64, (1, 3, 3), (1, 2, 2), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(64, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
        )
        
        self.encoder3 = nn.Sequential(
            ConvBlock(64, 128, (1, 3, 3), (1, 2, 2), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(128, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
        )
        
        self.encoder4 = nn.Sequential(
            ConvBlock(128, 256, (3, 3, 3), (1, 2, 2), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
        )
        
        self.encoder5 = nn.Sequential(
            ConvBlock(256, 320, (3, 3, 3), (2, 2, 2), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(320, 320, (3, 3, 3), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
        )
        
        self.encoder6 = nn.Sequential(
            ConvBlock(320, 320, (3, 3, 3), (2, 2, 2), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            ConvBlock(320, 320, (3, 3, 3), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
        )
        
        self.bottleneck = UNetBottleneck(320,320)

        self.decoder6 = nn.Sequential(
            ConvBlock(320, 320, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(320, 320, (2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        
        self.decoder5 = nn.Sequential(
            ConvBlock(320, 256, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(256, 256, (2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        
        self.decoder4 = nn.Sequential(
            ConvBlock(256, 128, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(128, 128, (2, 2, 2), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        
        self.decoder3 = nn.Sequential(
            ConvBlock(128, 64, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(64, 64, (2, 2, 2), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        
        self.decoder2 = nn.Sequential(
            ConvBlock(64, 32, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        
        self.decoder1 = nn.Sequential(
            ConvBlock(32, 32, (2, 2, 2), (1, 1, 1), (1, 1, 1), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True)),
            nn.Dropout3d(dropout_rate),
            nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(1, 1, 1), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )

        self.final_conv = ConvBlock(32, 1, (1, 1, 1), (1, 1, 1), (0, 0, 0), nn.InstanceNorm3d, nn.LeakyReLU(inplace=True))
    
    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        
        d6 = self.bottleneck(e6)

        d5 = self.decoder6(d6+e6)
        d4 = self.decoder5(d5 + e5)
        d3 = self.decoder4(d4 + e4)
        d2 = self.decoder3(d3 + e3)
        d1 = self.decoder2(d2 + e2)
        out = self.decoder1(d1 + e1)
        out = self.final_conv(out)
        out = out.permute(0, 1, 3, 4, 2)
        
        return out

# 测试模型结构
if __name__ == "__main__":
    model = CustomUNet3D()
    x = torch.randn(8, 1, 128, 128, 24)  # 输入大小(Batch, Channel, Depth, Height, Width)
    output = model(x)
    print(output.shape)  # 检查输出大小

'''