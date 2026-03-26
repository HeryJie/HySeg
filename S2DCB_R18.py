import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from network.models.Transformer_Eu import EU_MHSA

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, enc):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(enc) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x



class FAM1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(FAM1, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
                                  nn.Sigmoid())

    def forward(self, x):
        y = self.pool(x)
        x = self.conv(y)
        return x * y


class FAM2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(FAM2, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        b, c, _, _ = x.size()
        x = self.pool(x).view(b, c)
        x = self.fc(x).view(b, c, 1, 1)
        x = x * identity
        return x



class DownConnection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2):
        super(DownConnection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out



class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.channel_shuffle = ChannelShuffle(groups=4)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6(0.1)

    def forward(self, x, enc):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.channel_shuffle(x)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        x = fuse_weights[0] * self.pre_conv(enc) + fuse_weights[1] * x
        x = self.post_conv(x)

        shortcut = self.shortcut(x)

        pa = self.pa(x) * x
        ca = self.ca(x) * x

        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

########################################################################################################################
class SemanticPriorEncoder(nn.Module):
    def __init__(self, in_channels, num_classes, pyramid_channels=(32, 64, 128)):
        super().__init__()
        self.num_classes = num_classes
        # H//4 -> H//8
        self.enc1 = ConvBNReLU(in_channels, pyramid_channels[0], stride=2)
        # H//8 -> H//16
        self.enc2 = ConvBNReLU(pyramid_channels[0], pyramid_channels[1], stride=2)
        # H//16 -> H//32
        self.enc3 = ConvBNReLU(pyramid_channels[1], pyramid_channels[2], stride=2)

        self.proj1 = nn.Sequential(Conv(pyramid_channels[0], num_classes, kernel_size=1), nn.Softmax(dim=1))
        self.proj2 = nn.Sequential(Conv(pyramid_channels[1], num_classes, kernel_size=1), nn.Softmax(dim=1))
        self.proj3 = nn.Sequential(Conv(pyramid_channels[2], num_classes, kernel_size=1), nn.Softmax(dim=1))

    def forward(self, mf_out):
        prior_8 = self.enc1(mf_out)
        prior_16 = self.enc2(prior_8)
        prior_32 = self.enc3(prior_16)
        prob_8 = self.proj1(prior_8)
        prob_16 = self.proj2(prior_16)
        prob_32 = self.proj3(prior_32)
        return (prob_32, prob_16, prob_8)
########################################################################################################################
class DCBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, kernel_size=7, sigma=1.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.sigma = nn.Parameter(torch.tensor([sigma]), requires_grad=True)
        self.feature_conv = Conv(dim, dim, kernel_size=1)
        self.fusion_conv = Conv(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.attn = EU_MHSA(dim, dim, num_heads)
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()

    def _dynamic_crf_refinement(self, features, coarse_probs):
        B, C, H, W = coarse_probs.shape
        ################################################################################################################
        unfolded_probs = F.unfold(coarse_probs, kernel_size=self.kernel_size, padding=self.padding)
        unfolded_probs = unfolded_probs.view(B, C, self.kernel_size ** 2, H * W)

        center_pixel_prob = unfolded_probs[:, :, (self.kernel_size ** 2 - 1) // 2, :].unsqueeze(2)
        dist_sq = torch.sum((unfolded_probs - center_pixel_prob) ** 2, dim=1)

        affinity = torch.exp(-dist_sq / (2 * F.relu(self.sigma) ** 2 + 1e-8))
        affinity = F.softmax(affinity, dim=1)
        ################################################################################################################
        messages = self.feature_conv(features)

        unfolded_messages = F.unfold(messages, kernel_size=self.kernel_size, padding=self.padding)
        unfolded_messages = unfolded_messages.view(B, self.dim, self.kernel_size**2, H * W)

        aggregated_messages = torch.sum(unfolded_messages * affinity.unsqueeze(1), dim=2)
        aggregated_features = aggregated_messages.view(B, self.dim, H, W)
        ################################################################################################################

        refined_features = self.fusion_conv(aggregated_features)
        return refined_features

    def forward(self, x, coarse_probs):
        crf_refined_x = self._dynamic_crf_refinement(self.norm(x), coarse_probs)
        x = x + self.drop_path(crf_refined_x)

        attn_corrected_x = self.attn(x)
        x = x + self.drop_path(attn_corrected_x)
        return x
########################################################################################################################
class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = DCBlock(dim=decode_channels, num_heads=8)
        self.b3 = DCBlock(dim=decode_channels, num_heads=8)
        self.b2 = DCBlock(dim=decode_channels, num_heads=8)
        # self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        # self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        # self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.fam1 = FAM1(in_channels=encoder_channels[1])
        self.fam2 = FAM2(in_channels=encoder_channels[2])
        self.down12 = DownConnection(encoder_channels[0], encoder_channels[1])
        self.down231 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down232 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down34 = DownConnection(encoder_channels[2], encoder_channels[3])

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()
    def forward(self, enc1, enc2, enc3, enc4, h, w, coarse_priors_pyramid):
        prior_32, prior_16, prior_8 = coarse_priors_pyramid
        if self.training:
            f1_input = enc2 + self.down12(enc1)
            f2_input = enc3 + self.down231(enc2)
            f1_output = self.fam1(f1_input)
            f2_output = self.fam2(f2_input)
            input_3 = self.down232(f1_output) + f2_output
            input_4 = enc4 + self.down34(f2_output)

            x = self.b4(self.pre_conv(input_4), prior_32)
            h4 = self.up4(x)

            x = self.p3(x, input_3)
            x = self.b3(x, prior_16)
            h3 = self.up3(x)

            x = self.p2(x, enc2)
            x = self.b2(x, prior_8)
            h2 = x

            x = self.p1(x, enc1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            f1_input = enc2 + self.down12(enc1)
            f2_input = enc3 + self.down231(enc2)
            f1_output = self.fam1(f1_input)
            f2_output = self.fam2(f2_input)
            input_3 = self.down232(f1_output) + f2_output
            input_4 = enc4 + self.down34(f2_output)
            x = self.b4(self.pre_conv(input_4), prior_32)
            x = self.p3(x, input_3)
            x = self.b3(x, prior_16)

            x = self.p2(x, enc2)
            x = self.b2(x, prior_8)

            x = self.p1(x, enc1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class S2DCB_R18(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.semantic_encoder = SemanticPriorEncoder(in_channels=num_classes, num_classes=num_classes,
                                                     pyramid_channels=(32, 64, 128))
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x, mf_out):
        h, w = x.size()[-2:]
        coarse_priors_pyramid = self.semantic_encoder(mf_out)
        enc1, enc2, enc3, enc4 = self.backbone(x)
        z = [enc4, enc3, enc2]
        if self.training:
            x, ah = self.decoder(enc1, enc2, enc3, enc4, h, w, coarse_priors_pyramid)
            return x, ah
        else:
            x = self.decoder(enc1, enc2, enc3, enc4, h, w, coarse_priors_pyramid)
            return x, coarse_priors_pyramid