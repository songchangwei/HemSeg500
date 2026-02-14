import monai

def unetplusplus():
    unetplusplus = monai.networks.nets.BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32), 
        deep_supervision=False, 
        act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}), 
        norm=('instance', {'affine': True}), 
        bias=True, 
        dropout=0.5, 
        upsample='deconv'
    )
    return unetplusplus