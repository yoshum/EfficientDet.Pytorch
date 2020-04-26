import torch.nn as nn
import math
from .anchor_coder import AnchorCoder


MODEL_MAP = {
    "efficientdet-d0": "efficientnet-b0",
    "efficientdet-d1": "efficientnet-b1",
    "efficientdet-d2": "efficientnet-b2",
    "efficientdet-d3": "efficientnet-b3",
    "efficientdet-d4": "efficientnet-b4",
    "efficientdet-d5": "efficientnet-b5",
    "efficientdet-d6": "efficientnet-b6",
    "efficientdet-d7": "efficientnet-b6",
}


class DetectorModel(nn.Module):
    def __init__(self, backbone, neck, bbox_head, criterion, is_training=True):
        super(DetectorModel, self).__init__()
        self.backbone = backbone
        self.is_training = is_training
        self.neck = neck
        self.bbox_head = bbox_head

        self.coder = AnchorCoder()
        self.criterion = criterion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def forward(self, images):
        feature_map = self.backbone(images)
        feature_map = self.neck(feature_map[-5:])
        outs = self.bbox_head(feature_map)
        return outs

    def calculate_loss(self, inputs):
        inputs, annotations = inputs
        outs = self.forward(inputs)
        targets = self.coder.encode(annotations)
        return self.criterion(outs, targets)

    def predict(self, images):
        outs = self.forward(images)
        return self.coder.decode(outs)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x
