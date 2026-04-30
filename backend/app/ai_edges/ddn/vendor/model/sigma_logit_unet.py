import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):

        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        # encoder_channels: 576 384 192 96 [16]
        # decoder_channel: 256, 128, 64, 32, 16
        self.conv41 = DecoderBlock(encoder_channels[0], encoder_channels[1], decoder_channels[1])

        self.conv31 = DecoderBlock(encoder_channels[1], encoder_channels[2], decoder_channels[2])
        self.conv32 = DecoderBlock(decoder_channels[1], decoder_channels[2], decoder_channels[2])

        self.conv21 = DecoderBlock(encoder_channels[2], encoder_channels[3], decoder_channels[3])
        self.conv22 = DecoderBlock(decoder_channels[2], decoder_channels[3], decoder_channels[3])
        self.conv23 = DecoderBlock(decoder_channels[2], decoder_channels[3], decoder_channels[3])

        self.conv11 = DecoderBlock(encoder_channels[3], encoder_channels[4], decoder_channels[4])
        self.conv12 = DecoderBlock(decoder_channels[3], decoder_channels[4], decoder_channels[4])
        self.conv13 = DecoderBlock(decoder_channels[3], decoder_channels[4], decoder_channels[4])
        self.conv14 = DecoderBlock(decoder_channels[3], decoder_channels[4], decoder_channels[4])

    def forward(self, *features):
        # features = list(features[1:])  # remove first skip with same spatial resolution
        features = list(features)  # remove first skip with same spatial resolution
        features[0] = self.conv11(features[1], features[0])
        features[1] = self.conv21(features[2], features[1])
        features[2] = self.conv31(features[3], features[2])
        features[3] = self.conv41(features[4], features[3])

        features[0] = self.conv12(features[1], features[0])
        features[1] = self.conv22(features[2], features[1])
        features[2] = self.conv32(features[3], features[2])

        features[0] = self.conv13(features[1], features[0])
        features[1] = self.conv23(features[2], features[1])

        features[0] = self.conv14(features[1], features[0])
        return features[0]



def get_encoder(args):
    name = args.encoder
    if "CAFORMER-M36" == name.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=True)
    elif "CAFORMER-S18" == name.upper():
        from model.caformer import caformer_s18_384_in21ft1k
        encoder = caformer_s18_384_in21ft1k(pretrained=True)
    elif "DDN-M36" == name.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=False, Dulbrn=args.cfg.get("Dulbrn",16),cfg=args.cfg)
    elif "DDN-B36" == name.upper():
        from model.caformer import caformer_b36_384_in21ft1k
        encoder = caformer_b36_384_in21ft1k(pretrained=False, Dulbrn=args.cfg.get("Dulbrn",16),cfg=args.cfg)
    elif "DDN-S18" == name.upper():
        from model.caformer import caformer_s18_384_in21ft1k
        encoder = caformer_s18_384_in21ft1k(pretrained=False, Dulbrn=args.cfg.get("Dulbrn",16),cfg=args.cfg)
    elif "VGG" in name.upper():
        from model.VGG import VGG16_C
        encoder = VGG16_C(pretrain="model/vgg16.pth")
    elif "RESNET50" in name.upper():
        from model.resnet import resnet50_c
        encoder = resnet50_c()
    elif "RESNET101" in name.upper():
        from model.resnet import resnet101_c
        encoder = resnet101_c()
    else:
        raise Exception("uncorrect encoder")

    return encoder


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Mymodel(nn.Module):
    def __init__(self, args, classes=1,gaauss=True):
        super(Mymodel, self).__init__()

        self.gaauss = gaauss
        self.encoder_depth = 5
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.enh_size = (320, 480)
        self.decoder_use_batchnorm = True,

        self.decoder_attention_type = None

        self.encoder = get_encoder(args)
        encoder_channels = self.encoder.out_channels


        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=self.decoder_channels)



        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )

        if self.gaauss:
            self.decoder_1 = UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=self.decoder_channels,
            )
            self.segmentation_head_1 = SegmentationHead(
                in_channels=self.decoder_channels[-1],
                out_channels=classes,
                kernel_size=3,
            )


        self.args = args

    def forward(self, x):
        # VGG

        img_H, img_W = x.shape[2], x.shape[3]

        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        results = self.segmentation_head(decoder_output)
        results = interpolate(results, (img_H, img_W), mode="bilinear")
        if self.args.distribution == "beta":
            results = nn.Softplus()(results)

        if not self.gaauss:
            return results

        else:
            decoder_output_1 = self.decoder_1(*features)
            results_1 = self.segmentation_head_1(decoder_output_1)
            std = interpolate(results_1, (img_H, img_W),mode="bilinear")
            if self.args.distribution != "residual":
                std = nn.Softplus()(std)

            return results, std


