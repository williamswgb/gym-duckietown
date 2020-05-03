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

        self.conv1 = block2d(nn.Conv2d(3, 32, 3, stride=2))
        self.conv2 = block2d(nn.Conv2d(32, 32, 3, stride=2))
        self.conv3 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv4 = block2d(nn.Conv2d(32, 32, 3, stride=1))
        self.conv5 = block2d(nn.Conv2d(32, 32, 3, stride=1))

        self.dense = nn.Sequential(
          nn.Linear(32 * 23 * 33, 512),
          nn.ReLU(),
          nn.Dropout(0.3)
        )
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1)
        )

        self.flatten = nn.Flatten()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.linear(x)
        x = self.activation(x)

        return x