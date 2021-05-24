import torch as th
import torch.nn.functional as F

class GoNet(th.nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        self.conv1 = th.nn.Conv2d(3,  16, (3,3))
        self.conv2 = th.nn.Conv2d(16, 16, (3,3))
        self.conv3 = th.nn.Conv2d(16, 32, (3,3))
        self.conv4 = th.nn.Conv2d(32, 32, (3,3))
        self.lin = th.nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (3,3))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x,(3,3))
        x = x.reshape(-1,32)
        x = self.lin(x)
        X = F.softmax(x, dim=-1)
        return x

