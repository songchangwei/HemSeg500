import monai


def attentionunet():
    attentionunet = monai.networks.nets.AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256,512),
        strides=(2, 2, 2, 2),
        kernel_size = 3,
    )
    return attentionunet