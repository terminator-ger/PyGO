import torch as th
import torch.nn.functional as F
import pdb

def conv1x1(ipt, out, stride=1, dilation=1):
    return  th.nn.Sequential(
                th.nn.Conv2d(ipt, out, 
                    kernel_size=1, 
                    stride=stride, 
                    dilation=dilation,
                    bias=False),
                th.nn.BatchNorm2d(out)
            )

class ResBlock(th.nn.Module):
    def __init__(self, ipt, out, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        if stride == 1 and dilation == 1:
            padding = 1
        elif stride == 1 and dilation ==2:
            padding = 2
        elif stride == 2 and dilation == 1:
            padding = 1

        self.c1 = th.nn.Conv2d(ipt, out, (3,3), padding=1, bias=False)
        self.c2 = th.nn.Conv2d(out, out, (3,3), stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = th.nn.BatchNorm2d(out)
        self.bn2 = th.nn.BatchNorm2d(out)
        self.downsample = None
        self.downsample = conv1x1(ipt, out, stride=stride)

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

        self.block0 = ResBlock(16, 32, stride=2)        
        self.block1 = ResBlock(32, 64, stride=1, dilation=2)        
        self.block2 = ResBlock(64, 64, stride=1, dilation=2)        
        #self.block2 = ResBlock(64, 64)        
        self.block3 = ResBlock(64, 128, stride=1, dilation=2)        

        self.classifier = th.nn.Sequential(
            th.nn.AdaptiveMaxPool2d((1,1)),
            th.nn.Flatten(1),
            th.nn.Linear(128, 3), 
        )

    def forward(self, x):
        x = self.conv0(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

