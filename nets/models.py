import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import AmpNorm


def _zero_like_feature(feature):
    return feature.new_zeros(())


class RRSFLModelMixin:
    """Shared utilities for RRSFL stage embeddings and ICW regularization."""

    stage_dims = ()

    def _init_rrsfl(self, stage_dims, icw_groups=1):
        self.stage_dims = tuple(stage_dims)
        self.icw_groups = int(icw_groups)
        self.proj_heads = nn.ModuleList([nn.Linear(dim, dim) for dim in self.stage_dims])

    def set_icw_groups(self, group_num):
        self.icw_groups = max(1, int(group_num))

    def projection_parameters(self):
        return self.proj_heads.parameters()

    def task_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("proj_heads."):
                yield param

    def project_stage_vectors(self, stage_vectors):
        return [head(vec) for head, vec in zip(self.proj_heads, stage_vectors)]

    def stage_vectors_from_features(self, stage_features):
        return [F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1) for feat in stage_features]

    def icw_loss_from_features(self, stage_features):
        if not stage_features:
            return torch.tensor(0.0)

        total_loss = _zero_like_feature(stage_features[0])
        for feat in stage_features:
            total_loss = total_loss + self._single_stage_icw_loss(feat)
        return total_loss

    def _single_stage_icw_loss(self, feat, eps=1e-5):
        # Eq. 12-14 in the paper: group-wise relaxed whitening.
        batch, channels, height, width = feat.shape
        group_num = min(max(1, self.icw_groups), channels)
        channels_per_group = channels // group_num
        if channels_per_group < 2:
            return _zero_like_feature(feat)

        usable_channels = channels_per_group * group_num
        feat = feat[:, :usable_channels, :, :].contiguous()
        feat = feat.view(batch, group_num, channels_per_group, height * width)
        feat = feat - feat.mean(dim=-1, keepdim=True)
        feat = feat / (feat.std(dim=-1, keepdim=True, unbiased=False) + eps)

        denom = max(height * width - 1, 1)
        cov = torch.einsum("bgcn,bgdn->bgcd", feat, feat) / denom
        avg_cov = cov.mean(dim=1, keepdim=True)
        residual = cov - avg_cov

        eye = torch.eye(channels_per_group, device=feat.device, dtype=torch.bool)
        eye = eye.view(1, 1, channels_per_group, channels_per_group)
        psi = torch.where(
            residual > 0,
            torch.ones_like(cov),
            torch.zeros_like(cov),
        )
        psi = torch.where(eye, torch.ones_like(cov), psi).detach()

        per_group = torch.abs(cov - psi).mean(dim=(0, 2, 3))
        return per_group.sum()


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, **kwargs):
        super(_DenseLayer, self).__init__()
        self.add_module("bn1", nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module("bn2", nn.BatchNorm2d(bn_size * growth_rate, affine=False, track_running_stats=False))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, **kwargs):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, **kwargs):
        super(_Transition, self).__init__()
        self.add_module("bn", nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(RRSFLModelMixin, nn.Module):
    """DenseNet121 encoder with four RRSFL stage embeddings."""

    dense_stage_names = ("denseblock1", "denseblock2", "denseblock3", "denseblock4")

    def __init__(
        self,
        input_shape,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=2,
        use_amp_norm=False,
        icw_groups=1,
        **kwargs
    ):
        super(DenseNet, self).__init__()

        self.amp_norm = AmpNorm(input_shape=input_shape) if use_amp_norm else nn.Identity()
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("bn0", nn.BatchNorm2d(num_init_features, affine=False, track_running_stats=False)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        stage_dims = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            stage_dims.append(num_features)
            if i == 0:
                self.features.add_module("zero_padding", nn.ZeroPad2d(2))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("bn5", nn.BatchNorm2d(num_features, affine=False, track_running_stats=False))
        self.classifier = nn.Linear(num_features, num_classes)
        self._init_rrsfl(stage_dims=stage_dims, icw_groups=icw_groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias.data)

    def _forward_features(self, x):
        x = self.amp_norm(x)
        stage_features = []
        for name, module in self.features.named_children():
            x = module(x)
            if name in self.dense_stage_names:
                stage_features.append(x)
        return x, stage_features

    def extract_stage_vectors(self, x):
        _, stage_features = self._forward_features(x)
        return self.stage_vectors_from_features(stage_features)

    def forward(self, x, return_embeddings=True, return_icw=True):
        features, stage_features = self._forward_features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)

        embeddings = None
        if return_embeddings:
            embeddings = self.project_stage_vectors(self.stage_vectors_from_features(stage_features))
        icw_loss = self.icw_loss_from_features(stage_features) if return_icw else _zero_like_feature(logits)
        return logits, embeddings, icw_loss

    def stage_for_state_key(self, key):
        if key.startswith("proj_heads."):
            return None
        if key.startswith("amp_norm."):
            return 0
        if key.startswith(("features.conv0", "features.bn0", "features.relu0", "features.pool0",
                           "features.denseblock1", "features.zero_padding", "features.transition1")):
            return 0
        if key.startswith(("features.denseblock2", "features.transition2")):
            return 1
        if key.startswith(("features.denseblock3", "features.transition3")):
            return 2
        if key.startswith(("features.denseblock4", "features.bn5", "classifier")):
            return 3
        return 3


class UNet(RRSFLModelMixin, nn.Module):
    """U-Net with four encoder-stage RRSFL embeddings."""

    def __init__(
        self,
        input_shape,
        in_channels=3,
        out_channels=2,
        init_features=64,
        use_amp_norm=False,
        icw_groups=1,
    ):
        super(UNet, self).__init__()

        self.amp_norm = AmpNorm(input_shape=input_shape) if use_amp_norm else nn.Identity()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self._init_rrsfl(
            stage_dims=(features, features * 2, features * 4, features * 8),
            icw_groups=icw_groups,
        )

    def _forward_features(self, x):
        x = self.amp_norm(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        return (enc1, enc2, enc3, enc4)

    def extract_stage_vectors(self, x):
        stage_features = self._forward_features(x)
        return self.stage_vectors_from_features(stage_features)

    def forward(self, x, return_embeddings=True, return_icw=True):
        enc1, enc2, enc3, enc4 = self._forward_features(x)
        stage_features = (enc1, enc2, enc3, enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        logits = self.conv(dec1)

        embeddings = None
        if return_embeddings:
            embeddings = self.project_stage_vectors(self.stage_vectors_from_features(stage_features))
        icw_loss = self.icw_loss_from_features(stage_features) if return_icw else _zero_like_feature(logits)
        return logits, embeddings, icw_loss

    def stage_for_state_key(self, key):
        if key.startswith("proj_heads."):
            return None
        if key.startswith("amp_norm."):
            return 0
        if key.startswith(("encoder1", "pool1")):
            return 0
        if key.startswith(("encoder2", "pool2")):
            return 1
        if key.startswith(("encoder3", "pool3")):
            return 2
        return 3

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":
    exit()
