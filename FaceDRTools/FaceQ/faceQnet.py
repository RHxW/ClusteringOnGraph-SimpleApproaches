from torch import nn
import torchvision.models as models


class faceQnet(nn.Module):
    def __init__(self):
        super(faceQnet, self).__init__()

        self.backbone = models.resnet18(pretrained=False, num_classes=32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        return out
