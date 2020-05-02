from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        def block2d(layer):
            return nn.Sequential(
                layer,
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )

        self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = block2d(nn.Conv2d(3, 32, 3, stride=2))
        self.conv2 = block2d(nn.Conv2d(32, 32, 3, stride=2))
        self.conv3 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv4 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv5 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv6 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv7 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv8 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        
        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
          nn.Linear(32 * 17 * 27, 512),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(512, 1)
        )

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.activation(x)

        return x