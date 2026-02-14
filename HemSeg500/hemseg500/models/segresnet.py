import monai


def segresnet():
    segresnet = monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.2
    )
    return segresnet