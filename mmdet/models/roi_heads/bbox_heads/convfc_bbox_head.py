import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from .bbox_head import BBoxHead
from mmdet.core.bbox.iou_calculators import build_iou_calculator


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        self.wh_convs, self.wh_fcs, self.wh_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            #out_dim_reg = 2
            out_dim_center_and_wh = (2 if self.reg_class_agnostic else 2 * self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            self.fc_center = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_center_and_wh)
            self.fc_wh = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_center_and_wh)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class DecoupleShared2FCBBoxHead(ConvFCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DecoupleShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    def forward(self, reg_feats, cls_feats):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                reg_feats = conv(reg_feats)
                cls_feats = conv(cls_feats)
        #
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                reg_feats = self.avg_pool(reg_feats)
                cls_feats = self.avg_pool(cls_feats)

            reg_feats = reg_feats.flatten(1)
            cls_feats = cls_feats.flatten(1)

            for fc in self.shared_fcs:
                reg_feats = self.relu(fc(reg_feats))
                cls_feats = self.relu(fc(cls_feats))
        # separate branches

        for conv in self.cls_convs:
            cls_feats = conv(cls_feats)
        if cls_feats.dim() > 2:
            if self.with_avg_pool:
                cls_feats = self.avg_pool(cls_feats)
            cls_feats = cls_feats.flatten(1)
        for fc in self.cls_fcs:
            cls_feats = self.relu(fc(cls_feats))

        for conv in self.reg_convs:
            reg_feats = conv(reg_feats)
        if reg_feats.dim() > 2:
            if self.with_avg_pool:
                reg_feats = self.avg_pool(reg_feats)
            reg_feats = reg_feats.flatten(1)
        for fc in self.reg_fcs:
            reg_feats = self.relu(fc(reg_feats))

        cls_score = self.fc_cls(cls_feats) if self.with_cls else None
        bbox_pred = self.fc_reg(reg_feats) if self.with_reg else None
        return cls_score, bbox_pred

@HEADS.register_module()
class DecoupleRefinedShared2FCBBoxHead(ConvFCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DecoupleRefinedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.delta_xywh = DeltaXYWHBBoxCoder()

    def forward(self, center_feats, cls_feats, wh_fea, rois, rois_extractor):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                center_feats = conv(center_feats)
                cls_feats = conv(cls_feats)
        #
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                center_feats = self.avg_pool(center_feats)
                cls_feats = self.avg_pool(cls_feats)

            center_feats = center_feats.flatten(1)
            cls_feats = cls_feats.flatten(1)

            for fc in self.shared_fcs:
                center_feats = self.relu(fc(center_feats))
                cls_feats = self.relu(fc(cls_feats))
        # separate branches

        for conv in self.cls_convs:
            cls_feats = conv(cls_feats)
        if cls_feats.dim() > 2:
            if self.with_avg_pool:
                cls_feats = self.avg_pool(cls_feats)
            cls_feats = cls_feats.flatten(1)
        for fc in self.cls_fcs:
            cls_feats = self.relu(fc(cls_feats))

        for conv in self.reg_convs:
            center_feats = conv(center_feats)
        if center_feats.dim() > 2:
            if self.with_avg_pool:
                center_feats = self.avg_pool(center_feats)
            center_feats = center_feats.flatten(1)
        for fc in self.reg_fcs:
            center_feats = self.relu(fc(center_feats))

        cls_score = self.fc_cls(cls_feats) if self.with_cls else None
        center_pred = self.fc_center(center_feats) if self.with_reg else None
        #start refine proposal and then predict the wh offset
        if self.reg_class_agnostic:
            tmp_tensor = torch.zeros([center_pred.size(0), 4], device=center_feats.device)
            tmp_tensor[..., :2] = center_pred
            decode_rois = self.delta_xywh.decode(rois[..., 1:], tmp_tensor)
            tmp_rois = rois.clone()
            tmp_rois[..., 1:] = decode_rois
            wh_feats = rois_extractor(wh_fea[:rois_extractor.num_inputs], tmp_rois)
        else:
            # cls_max = torch.max(cls_score, dim=1, keepdim=True)[1].unsqueeze(2).expand(-1, 1, 2)
            # tmp_tensor = center_pred.view(center_pred.size(0), self.num_classes, -1)
            # target_center = torch.gather(tmp_tensor, dim=1, index=cls_max)
            row_index = torch.arange(center_pred.size(0)).to(center_feats.device)
            cls_max = torch.max(cls_score[..., :self.num_classes], dim=1)[1]
            tmp_center = center_pred.view(center_pred.size(0), self.num_classes, -1)
            target_center = tmp_center[row_index, cls_max, ...]
            tmp_tensor = torch.zeros([center_pred.size(0), 4], device=center_feats.device)
            tmp_tensor[..., :2] = target_center
            decode_rois = self.delta_xywh.decode(rois[..., 1:], tmp_tensor)
            tmp_rois = rois.clone()
            tmp_rois[..., 1:] = decode_rois
            wh_feats = rois_extractor(wh_fea[:rois_extractor.num_inputs], tmp_rois)

        #wh regress
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                wh_feats = conv(wh_feats)
        #
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                wh_feats = self.avg_pool(wh_feats)

            wh_feats = wh_feats.flatten(1)

            for fc in self.shared_fcs:
                wh_feats = self.relu(fc(wh_feats))
        for conv in self.reg_convs:
            wh_feats = conv(wh_feats)
        if wh_feats.dim() > 2:
            if self.with_avg_pool:
                wh_feats = self.avg_pool(wh_feats)
            wh_feats = wh_feats.flatten(1)
        for fc in self.reg_fcs:
            wh_feats = self.relu(fc(wh_feats))
        wh_pred = self.fc_wh(wh_feats) if self.with_reg else None
        if self.reg_class_agnostic:
            bbox_pred = torch.cat([center_pred, wh_pred], dim=1)
        else:
            bbox_pred = torch.cat([center_pred.view(center_pred.size(0), self.num_classes, -1), wh_pred.view(wh_pred.size(0), self.num_classes, -1)], dim=2).view(center_pred.size(0), -1)
        return cls_score, bbox_pred
    

@HEADS.register_module()
class DecoupleThreeShared2FCBBoxHead(ConvFCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DecoupleThreeShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    def forward(self, center_feats, wh_feats, cls_feats,):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                center_feats = conv(center_feats)
                wh_feats = conv(wh_feats)
                cls_feats = conv(cls_feats)
        #
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                center_feats = self.avg_pool(center_feats)
                wh_feats = self.avg_pool(wh_feats)
                cls_feats = self.avg_pool(cls_feats)

            center_feats = center_feats.flatten(1)
            cls_feats = cls_feats.flatten(1)
            wh_feats = wh_feats.flatten(1)

            for fc in self.shared_fcs:
                center_feats = self.relu(fc(center_feats))
                wh_feats = self.relu(fc(wh_feats))
                cls_feats = self.relu(fc(cls_feats))
        # separate branches

        for conv in self.cls_convs:
            cls_feats = conv(cls_feats)
        if cls_feats.dim() > 2:
            if self.with_avg_pool:
                cls_feats = self.avg_pool(cls_feats)
            cls_feats = cls_feats.flatten(1)
        for fc in self.cls_fcs:
            cls_feats = self.relu(fc(cls_feats))

        for conv in self.reg_convs:
            center_feats = conv(center_feats)
        if center_feats.dim() > 2:
            if self.with_avg_pool:
                center_feats = self.avg_pool(center_feats)
            center_feats = center_feats.flatten(1)
        for fc in self.reg_fcs:
            center_feats = self.relu(fc(center_feats))
            
        for conv in self.reg_convs:
            wh_feats = conv(wh_feats)
        if wh_feats.dim() > 2:
            if self.with_avg_pool:
                wh_feats = self.avg_pool(wh_feats)
            wh_feats = wh_feats.flatten(1)
        for fc in self.reg_fcs:
            wh_feats = self.relu(fc(wh_feats))

        cls_score = self.fc_cls(cls_feats) if self.with_cls else None
        center_pred = self.fc_center(center_feats) if self.with_reg else None
        wh_pred = self.fc_wh(wh_feats) if self.with_reg else None
        if self.reg_class_agnostic:
            bbox_pred = torch.cat([center_pred, wh_pred], dim=1)
        else:
            bbox_pred = torch.cat([center_pred.view(center_pred.size(0), self.num_classes, -1), wh_pred.view(wh_pred.size(0), self.num_classes, -1)], dim=2).view(center_pred.size(0), -1)
        return cls_score, bbox_pred

@HEADS.register_module()
class DecoupleCenterWHHead(ConvFCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DecoupleCenterWHHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        self.cls_fcs = nn.ModuleList([
            nn.Linear(12544, 1024),
            nn.Linear(1024, 1024)
        ])
        self.center_fcs = nn.ModuleList([
            nn.Linear(12544, 1024),
            nn.Linear(1024, 1024)
        ])
        self.wh_fcs = nn.ModuleList([
            nn.Linear(12544, 1024),
            nn.Linear(1024, 1024)
        ])
        self.fc_center = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=2)
        self.fc_wh = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=2)
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

    def forward(self, bbox_center_feats, bbox_wh_feats, bbox_cls_feats):
        bbox_center_feats = bbox_center_feats.flatten(1)
        bbox_wh_feats = bbox_wh_feats.flatten(1)
        bbox_cls_feats = bbox_cls_feats.flatten(1)
        for index in range(len(self.center_fcs)):
            bbox_center_feats = self.relu(self.center_fcs[index](bbox_center_feats))
            bbox_wh_feats = self.relu(self.wh_fcs[index](bbox_wh_feats))
            bbox_cls_feats = self.relu(self.cls_fcs[index](bbox_cls_feats))
        cls_score = self.fc_cls(bbox_cls_feats) if self.with_cls else None
        center_pred = self.fc_center(bbox_center_feats) if self.with_reg else None
        wh_pred = self.fc_wh(bbox_wh_feats) if self.with_reg else None
        bbox_pred = torch.cat([center_pred, wh_pred], dim=1)
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]
                pos_target = bbox_targets[pos_inds.type(torch.bool)]
                decode_pred_bbox = self.bbox_coder.decode(rois[pos_inds.type(torch.bool), 1:], pos_bbox_pred)
                decode_target_bbox = self.bbox_coder.decode(rois[pos_inds.type(torch.bool), 1:], pos_target)
                overlaps = self.iou_calculator(decode_target_bbox, decode_pred_bbox, is_aligned=True)
                iou_adaptive_weight = torch.cat([(1-overlaps).unsqueeze(-1), (1-overlaps).unsqueeze(-1), overlaps.unsqueeze(-1), overlaps.unsqueeze(-1)], dim=-1)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    pos_target,
                    iou_adaptive_weight,
                    avg_factor=bbox_targets.size(0)//2,
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

