import torch as th
import torch.nn.functional as F
import pdb

def conv1x1(ipt, out, stride):
    return  th.nn.Sequential(
                th.nn.Conv2d(ipt, out, kernel_size=1, stride=stride, bias=False),
                th.nn.BatchNorm2d(out)
            )

class ResBlock(th.nn.Module):
    def __init__(self, ipt, out, stride):
        super(ResBlock, self).__init__()
        padding = 1
        self.c1 = th.nn.Conv2d(ipt, out, (3,3), padding=padding)
        self.c2 = th.nn.Conv2d(out, out, (3,3), stride=stride, padding=padding)
        self.bn1 = th.nn.BatchNorm2d(out)
        self.bn2 = th.nn.BatchNorm2d(out)
        self.downsample = None
        self.downsample = conv1x1(ipt, out, stride)

    def forward(self, x):
        identity = x
        out = self.c1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.c2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out
 


class GoNet(th.nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        
        self.conv0 = th.nn.Conv2d(1,  16, (3,3), padding=1)

        self.block0 = ResBlock(16, 32, 1)        
        self.block1 = ResBlock(32, 32, 2)        
        self.block2 = ResBlock(32, 64, 1)        
        self.block3 = ResBlock(64, 64, 2)        

        self.classifier = th.nn.Sequential(
            th.nn.Linear(64 * 8 * 8, 3), 
            th.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv0(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

