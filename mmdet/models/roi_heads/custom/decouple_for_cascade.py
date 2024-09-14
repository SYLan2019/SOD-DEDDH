import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

class MultipleScaleFeature(nn.Module):
    def __init__(self, inter_num=3, in_channels=256):
        super(MultipleScaleFeature, self).__init__()
        self.inter_num = inter_num
        self.in_channels = in_channels
        self.mix_block = self.builder_inter_conv()

    def builder_inter_conv(self):
        tmp_conv = []
        for index in range(self.inter_num):
            tmp_conv.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=self.in_channels, out_channels=self.in_channels,
                                                 kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
            )
        return nn.ModuleList(tmp_conv)

    @auto_fp16()
    def forward(self, x):
        tmp = []
        for index, block in enumerate(self.mix_block):
            input = (x if index == 0 else tmp[-1])
            tmp_out = block(input) + x
            tmp.append(tmp_out)
        out = torch.cat(tmp, dim=1)
        return out

class DynamicSpatialSelect(nn.Module):
    def __init__(self, outchannels=256, head_num=2, cat_num=3):
        super(DynamicSpatialSelect, self).__init__()
        self.conv1 = nn.Conv2d(2, head_num, 5, 1, 2)
        self.act = nn.Sigmoid()
        self.reduce_conv1 = nn.Conv2d(in_channels=outchannels * cat_num, out_channels=outchannels, kernel_size=1)
        self.reduce_conv2 = nn.Conv2d(in_channels=outchannels * cat_num, out_channels=outchannels, kernel_size=1)

    @auto_fp16()
    def forward(self, x):
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        mean_x = torch.mean(x, dim=1, keepdim=True)
        cat_fea = torch.cat([max_x, mean_x], dim=1)
        att = self.act(self.conv1(cat_fea))
        x1 = att[:, 0, ...].unsqueeze(1) * x
        x2 = att[:, 1, ...].unsqueeze(1) * x
        #reduce channel
        x1 = self.reduce_conv1(x1)
        x2 = self.reduce_conv1(x2)
        return x1, x2

class DynamicChannelSelect(nn.Module):
    def __init__(self, in_channels=256, branch_num=3):
        super(DynamicChannelSelect, self).__init__()
        self.inchannels = in_channels
        self.branch_num = branch_num
        input_dim = in_channels * branch_num
        self.adap_max_pool = nn.AdaptiveMaxPool2d(1)
        self.adap_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim * 2)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        max_fea = self.adap_max_pool(x).view(b, c)
        avg_fea = self.adap_avg_pool(x).view(b, c)
        sum_max_avg_fea = max_fea + avg_fea
        all_att = self.act(self.linear1(sum_max_avg_fea))
        all_att = all_att.reshape(b, 2, -1).permute(1, 0, 2).contiguous()
        reg_att, cls_att = all_att[0], all_att[1]
        reg_att_list = torch.chunk(reg_att, self.branch_num, dim=1)
        cls_att_list = torch.chunk(cls_att, self.branch_num, dim=1)
        reg_att = torch.stack(reg_att_list, dim=1)
        cls_att = torch.stack(cls_att_list, dim=1)
        reg_att = F.softmax(reg_att, dim=1)
        cls_att = F.softmax(cls_att, dim=1)
        fea_list = torch.chunk(x, self.branch_num, dim=1)
        fea = torch.stack(fea_list, dim=1)
        reg_fea = reg_att.unsqueeze(-1).unsqueeze(-1) * fea
        cls_fea = cls_att.unsqueeze(-1).unsqueeze(-1) * fea
        reg_fea = torch.sum(reg_fea, dim=1)
        cls_fea = torch.sum(cls_fea, dim=1)
        return reg_fea, cls_fea

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=256):
        super(SpatialAttention, self).__init__()
        self.mix = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        max_fea = torch.max(x, dim=1, keepdim=True)[0]
        mean_fea = torch.mean(x, dim=1, keepdim=True)
        cat_fea = torch.cat([max_fea, mean_fea], dim=1)
        spatial_att = self.act(self.mix(cat_fea))
        out = spatial_att * x
        return out

class GlobalRefine(BaseModule):
    def __init__(self, in_channels=256):
        super(GlobalRefine, self).__init__()
        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        tmp = q @ torch.transpose(k, -2, -1)
        score = F.softmax(tmp, dim=-1)
        out = score @ v
        return out

class WHAttention(nn.Module):
    def __init__(self):
        super(WHAttention, self).__init__()
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Conv1d(1, 1, 3, 1, 1)
        self.act = nn.ReLU()
        self.linear2 = nn.Conv1d(1, 1, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_fea = x.clone().permute((0, 2, 1, 3)).contiguous()
        w_fea = x.clone().permute((0, 3, 2, 1)).contiguous()
        h_fea = self.adap_pool(h_fea).view(B, H)
        w_fea = self.adap_pool(w_fea).view(B, W)
        cat_fea = torch.cat([h_fea, w_fea], dim=1).unsqueeze(1)
        out = self.linear2(self.act(self.linear1(cat_fea))).view((B, H + W))
        h_att = out[..., :H]
        w_att = out[..., H:]
        return h_att.unsqueeze(1).unsqueeze(3) * x * w_att.unsqueeze(1).unsqueeze(2)

class SpatialAndChannelGlobalEnhance(nn.Module):
    def __init__(self, inchanels=256):
        super(SpatialAndChannelGlobalEnhance, self).__init__()
        self.inchannels = inchanels
        self.v_conv = nn.Conv2d(in_channels=inchanels, out_channels=inchanels, kernel_size=1, stride=1)
        self.k_conv = nn.Conv2d(in_channels=inchanels, out_channels=1, kernel_size=1, stride=1)
        self.adap_max_pool = nn.AdaptiveMaxPool2d(1)
        self.adap_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 1)
        self.conv2 = nn.Conv2d(2, 1, 1)
        self.conv3 = nn.Conv2d(in_channels=inchanels, out_channels=1, kernel_size=1)
        # self.conv4 = nn.Conv2d(in_channels=inchanels, out_channels=inchanels//2, kernel_size=1)
        # self.act1 = nn.ReLU()
        # self.conv5 = nn.Conv2d(in_channels=inchanels//2, out_channels=inchanels, kernel_size=1)
        #self.conv3 = nn.Conv2d(inchanels, )


    def forward(self, x):
        b, c, h, w = x.shape
        tmp = self.v_conv(x)
        v = tmp.view(b, c, -1)
        #v = self.v_conv(x).view(b, c, -1)
        #tmp = self.k_conv(x)
        k = self.k_conv(x).view(b, -1, 1)
        k = F.softmax(k, dim=1)
        channel_fea = (v @ k).unsqueeze(-1)
        #channel_fea = self.conv5(self.act1(self.conv4(channel_fea)))
        h_fea = x.clone().permute(0, 2, 1, 3).contiguous()
        w_fea = x.clone().permute(0, 3, 1, 2).contiguous()
        h_max_fea = self.adap_max_pool(h_fea).permute(0, 2, 1, 3).contiguous()
        h_avg_fea = self.adap_avg_pool(h_fea).permute(0, 2, 1, 3).contiguous()
        w_max_fea = self.adap_max_pool(w_fea).permute(0, 2, 3, 1).contiguous()
        w_avg_fea = self.adap_avg_pool(w_fea).permute(0, 2, 3, 1).contiguous()
        h_all_fea = torch.cat([h_max_fea, h_avg_fea], dim=1)
        w_all_fea = torch.cat([w_max_fea, w_avg_fea], dim=1)
        h_att = self.conv1(h_all_fea)
        #h_att = F.softmax(h_att, dim=2)
        h_att = self.act(h_att)
        w_att = self.conv2(w_all_fea)
        #w_att = F.softmax(w_att, dim=3)
        w_att = self.act(w_att)
        spatial_fea = h_att * tmp + w_att * tmp
        spatial_fea = self.act(self.conv3(spatial_fea))
        out = spatial_fea * channel_fea
        return out + x
    
class Mlp(nn.Module):
    def __init__(self, in_features=256, hidden_features=None, out_features=256, act_layer=nn.GELU, drop=0., group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group > 0:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=group)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if group > 0:
            self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=group)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.group = group

    def forward(self, x):
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        return x

class DecoupleTaskInteraction(nn.Module):
    def __init__(self, input_dim=256, head_num=3, with_position=False):
        super(DecoupleTaskInteraction, self).__init__()
        self.with_position = with_position
        self.mix = nn.Conv2d(in_channels=head_num * input_dim, out_channels=input_dim, kernel_size=1)
        self.norm = nn.LayerNorm(input_dim)
        self.q_conv1 = nn.Linear(input_dim, input_dim,)
        self.q_conv2 = nn.Linear(input_dim, input_dim)
        self.q_conv3 = nn.Linear(input_dim, input_dim)
        self.k_conv = nn.Linear(input_dim, input_dim)
        self.v_conv = nn.Linear(input_dim, input_dim)
        self.center_mlp = Mlp()
        self.wh_mlp = Mlp()
        self.cls_mlp = Mlp()

    def forward(self, center_fea, wh_fea, cls_fea):
        b, c, h, w = center_fea.shape
        cat_fea = torch.cat([center_fea, wh_fea, cls_fea], dim=1)
        mix_fea = self.mix(cat_fea)
        center_fea = center_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        wh_fea = wh_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        cls_fea = cls_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        mix_fea = mix_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        center_tmp, wh_tmp, cls_tmp, mix_fea = self.norm(center_fea), self.norm(wh_fea), self.norm(cls_fea), self.norm(mix_fea)
        center_q = self.q_conv1(center_tmp)
        wh_q = self.q_conv2(wh_tmp)
        cls_q = self.q_conv3(cls_tmp)
        k_fea = self.k_conv(mix_fea)
        v_fea = self.v_conv(mix_fea)
        center_score = F.softmax(center_q @ k_fea.transpose(-2, -1), dim=2)
        wh_score = F.softmax(wh_q @ k_fea.transpose(-2, -1), dim=2)
        cls_score = F.softmax(cls_q @ k_fea.transpose(-2, -1), dim=2)
        center_out, wh_out, cls_out = center_score @ v_fea, wh_score @ v_fea, cls_score @ v_fea
        center_fea, wh_fea, cls_fea = self.norm(center_fea + center_out), self.norm(wh_fea + wh_out), self.norm(cls_fea + cls_out)
        center_fea, wh_fea, cls_fea = self.norm(center_fea + self.center_mlp(center_fea)), self.norm(wh_fea + self.wh_mlp(wh_fea)), self.norm(cls_fea + self.cls_mlp(cls_fea))
        center_fea, wh_fea, cls_fea = center_fea.permute(0, 2, 1).contiguous().view(b, c, h, w), wh_fea.permute(0, 2, 1).contiguous().view(b, c, h, w), cls_fea.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return center_fea, wh_fea, cls_fea
