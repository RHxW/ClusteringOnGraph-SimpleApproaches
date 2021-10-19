from torch import nn
import torchvision.models as models


class GenderClsNetwork(nn.Module):
    def __init__(self):
        super(GenderClsNetwork, self).__init__()

        self.backbone = models.resnet18(pretrained=False, num_classes=32)
        # self.backbone = models.resnet34(pretrained=False, num_classes=32)
        self.fc = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        out = self.softmax(x)
        return out