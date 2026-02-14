import monai

def vnet():
        vnet = monai.networks.nets.VNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
        )
        return vnet