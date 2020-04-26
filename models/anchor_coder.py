import torch
from torchvision.ops import nms

import numpy as np

from .utils import calculate_iou


class AnchorGenerator(object):
    def __init__(
        self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None
    ):
        super(AnchorGenerator, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def __call__(self, image_size):

        image_size = np.array(image_size)
        image_size = [
            (image_size + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels
        ]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
            )
            shifted_anchors = shift(image_size[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32))


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2)
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


class AnchorCoder(object):
    def __init__(
        self,
        image_size,
        n_fg_classes,
        anchors=None,
        iou_threshold=0.5,
        negative_iou_threshold=0.4,
        score_threshold=0.01,
        mean=None,
        std=None,
    ):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.n_fg_classes = n_fg_classes
        if anchors is None:
            self.anchors = AnchorGenerator()(image_size)
        else:
            self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.score_threshold = score_threshold
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            )
        else:
            self.std = std

    def encode(self, annotation):
        cls_targets = []
        reg_targets = []

        for anno in annotation:
            cls_target, reg_target = self.encode_single(anno)
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)

        return torch.stack(cls_targets), torch.stack(reg_targets)

    def encode_single(self, annotation):
        anchors = self.anchors.copy()
        assigned_annotations, positive_indices, negative_indices = self._assign(
            anchors, annotation
        )
        # positive_anchors = self.anchors.copy()[positive_indices, :]

        cls_targets = torch.ones((len(anchors), self.n_fg_classes)) * -1
        cls_targets = cls_targets.cuda()
        cls_targets[negative_indices, :] = 0
        cls_targets[positive_indices, :] = 0
        cls_targets[
            positive_indices, assigned_annotations[positive_indices, 4].long()
        ] = 1

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

        gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
        gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

        # clip widths to 1
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)

        targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
        targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
        targets_dw = torch.log(gt_widths / anchor_widths)
        targets_dh = torch.log(gt_heights / anchor_heights)

        reg_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
        reg_targets = reg_targets.t()

        reg_targets = reg_targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

        return cls_targets, reg_targets

    def decode(self, outs):
        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        anchors = self.anchors.copy()
        transformed_anchors = self._regress_boxes(anchors, regression)
        transformed_anchors = self._clip_boxes(transformed_anchors)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.score_threshold)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            print("No boxes to NMS")
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx = nms(
            transformed_anchors[0, :, :],
            scores[0, :, 0],
            iou_threshold=self.iou_threshold,
        )
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def _assign(self, anchors, bbox_annotation):
        IoU = calculate_iou(anchors[0, :, :], bbox_annotation[:, :4])
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

        negative_anchors = torch.lt(IoU_max, self.negative_iou_threshold)
        positive_anchors = torch.ge(IoU_max, self.iou_threshold)

        assigned_annotations = bbox_annotation[IoU_argmax, :]

        return assigned_annotations, positive_anchors, negative_anchors

    def _clip_boxes(self, boxes):
        height, width = self.image_size

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

    def _regress_boxes(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes
