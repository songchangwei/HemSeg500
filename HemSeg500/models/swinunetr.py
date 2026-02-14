import monai

def swinunetr():
        swinunetr = monai.networks.nets.SwinUNETR(
                img_size=(64,64,32),
                in_channels=1,
                out_channels=1, 
                feature_size=48,
        )
        return swinunetr
