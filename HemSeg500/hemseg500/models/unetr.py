import monai

def unetr():
    unetr = monai.networks.nets.UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(256, 256, 16),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        dropout_rate=0.0,
    )
    return unetr