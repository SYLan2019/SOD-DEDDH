import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from ..builder import NECKS
import torch.nn as nn
import torch

class Sobel2D(BaseModule):
    def __init__(self, in_channels=256):
        super(Sobel2D, self).__init__()
        self.x_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.y_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        x_sobel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1).clone()
        y_sobel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1).clone()
        self.x_conv.weight = nn.Parameter(x_sobel)
        self.y_conv.weight = nn.Parameter(y_sobel)
        self.act = nn.ReLU()

    @auto_fp16()
    def forward(self, x):
        x_edge = self.act(self.x_conv(x))
        y_edge = self.act(self.y_conv(x))
        out = torch.sqrt(x_edge**2 + y_edge**2)
        return out

class NoiseRemoveAttention(BaseModule):
    def __init__(self, in_channels=1024):
        super(NoiseRemoveAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.mix = nn.Conv1d(1, 1, 3, 1, 1)
        # self.act1 = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 7, 1, 3)
        self.act2 = nn.Sigmoid()
        self.gConvMix = nn.Conv2d(in_channels * 2, in_channels * 2, 7, 1, 3, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels//4, 1, 1, 0)
        self.act3 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels//4, in_channels, 1, 1, 0)
        self.act4 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(in_channels * 2, in_channels, 1, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        # tmp = x.permute(0, 3, 1, 2)
        # avg_w = self.pool1(tmp)
        # tmp = x.permute(0, 2, 1, 3)
        # avg_h = self.pool2(tmp)
        # avg_w = avg_w.squeeze()
        # avg_h = avg_h.squeeze()
        # if B == 1:
        #     avg_h = avg_h.unsqueeze(0)
        #     avg_w = avg_w.unsqueeze(0)
        # w_len = avg_w.shape[-1]
        # cat_feature = torch.cat([avg_w, avg_h], dim=1).unsqueeze(1)
        # cat_feature = self.act1(self.mix(cat_feature))
        # w_att = cat_feature[..., 0:w_len].unsqueeze(2)
        # h_att = cat_feature[..., w_len:].unsqueeze(3)
        max_feature = torch.max(x, dim=1, keepdim=True)[0]
        avg_feature = torch.mean(x, dim=1, keepdim=True)
        cat_avg_mean = torch.cat([max_feature, avg_feature], dim=1)
        spatial_att = self.act2(self.conv1(cat_avg_mean))
        avg_channel_feature = self.pool(x)
        channel_att = self.conv3(self.act3(self.conv2(avg_channel_feature)))
        #coarse_att = w_att * h_att + spatial_att + channel_att
        coarse_att = spatial_att + channel_att
        cat_att_content = torch.stack(([coarse_att, x]), dim=1).permute(0, 2, 1, 3, 4).contiguous().view(B, -1, H, W)
        refine_att = self.act4(self.conv4(self.gConvMix(cat_att_content)))
        return x * refine_att

class DWTRefineBlock(BaseModule):
    def __init__(self, in_channels=None):
        super(DWTRefineBlock, self).__init__()
        self.dwt_2D = DWT_2D('haar')
        self.idwt_2D = IDWT_2D('haar')
        self.sobel = nn.ModuleList([Sobel2D(in_channels) for _ in range(3)])
        self.mix = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.att = nn.ModuleList([NoiseRemoveAttention(in_channels) for _ in range(3)])

    def forward_feature(self, x):
        LL, LH, HL, HH = self.dwt_2D(x)
        high_f = [LH, HL, HH]
        for index, item in enumerate(high_f):
            out = self.mix(item + self.sobel[index](item)).clone()
            out = self.att[index](out)
            if index==0:
                LH = out
            elif index==1:
                HL = out
            else:
                HH = out
        out = self.idwt_2D(LL, LH, HL, HH)
        return out

    def forward(self, x):
        out = self.forward_feature(x)
        return out


@NECKS.register_module()
class DWTFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DWTFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.dwt_refine = nn.ModuleList([DWTRefineBlock(in_channel) for in_channel in in_channels])
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        #DWT精炼
        refined_inputs = [
           dwt(inputs[index]) for index, dwt in enumerate(self.dwt_refine)
        ]
        # build laterals
        laterals = [
            lateral_conv(refined_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
