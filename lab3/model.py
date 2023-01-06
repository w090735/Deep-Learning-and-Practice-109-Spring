from utils import *

class EEGNet(nn.Module):
    def __init__(self, activate=nn.ELU(alpha=1.0)):
        super(EEGNet, self).__init__()
        # firstconv
        self.firstconv = nn.Sequential(
            # number of filters = 16
            # (750 + 25*2) - 51 + 1 = 750
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            # [16, 2, 750]
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ) 
        # depthwiseConv
        self.depthwiseConv = nn.Sequential(
            # number of filters = 32
            # 2 - 2 + 1 = 1
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            # [32, 1, 750]
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 750 / 4 = 187
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            # [32, 1, 187]
            nn.Dropout(p=0.25)
        )
        # separableConv
        self.separableConv = nn.Sequential(
            # number of filters = 32
            # (187 + 7*2) - 15 + 1 = 187 
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            # [32, 1, 187]
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 187 / 8 = 23
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            # [32, 1, 23]
            nn.Dropout(p=0.25)
        )
        # classify
        self.classify = nn.Sequential(
            # 32*23 = 736
            # class = 2
            nn.Linear(in_features=736, out_features=2, bias=True)
            # [736]
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # torch.Size([64, 32, 1, 23])
        # 32x23 = 736
        # reshape to (64, 736)
        x = x.view(x.shape[0],-1)
        output = self.classify(x)
        return output

class DeepConvNet(nn.Module):
    def __init__(self, activate=nn.ELU(alpha=1.0)):
        super(DeepConvNet, self).__init__()
        # stride 3 2  1
        # acc  82 84 77
        # Conv-Pool-Block 1
        # bias True False
        # acc   84   83.9
        self.ConvPoolBlock1 = nn.Sequential(
            # number of filters = 25
            # 750 / 2 - 2 = 373
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 2), padding=0, bias=True),
            # [25, 2, 373]
            # number of filters = 25
            # 2 - 2 + 1 = 1
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), groups=25, bias=True),
            # [25, 1, 373]
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 373 - 2 + 1 = 372
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0),
            # [25, 1, 372]
            nn.Dropout(p=0.35)
        ) 
        # Conv-Pool Block 2
        self.ConvPoolBlock2 = nn.Sequential(
            # number of filters = 50
            # 372 / 2 - 2 = 184
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 2), padding=0, bias=True),
            # [50, 1, 184]
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 184 - 2 + 1 = 183
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0),
            # [50, 1, 183]
            nn.Dropout(p=0.35)
        )
        # Conv-Pool-Block 3
        self.ConvPoolBlock3 = nn.Sequential(
            # number of filters = 100
            # 183 / 2 - 1 = 90
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 2), padding=0, bias=True),
            # [100, 1, 90]
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 90 - 2 + 1 = 89
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0),
            # [100, 1, 89]
            nn.Dropout(p=0.35)
        )
        # Conv-Pool-Block 4
        self.ConvPoolBlock4 = nn.Sequential(
            # number of filters = 200
            # 89 / 2 - 2 = 43
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 2), padding=0, bias=True),
            # [200, 1, 43]
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate,
            # 43 - 2 + 1 = 42
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0),
            # [200, 1, 42]
            nn.Dropout(p=0.35)
        )
        # Classification Layer
        self.ClassificationLayer = nn.Sequential(
            # 200*42 = 8400
            # class = 2
            nn.Linear(in_features=8400, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.ConvPoolBlock1(x)
        x = self.ConvPoolBlock2(x)
        x = self.ConvPoolBlock3(x)
        x = self.ConvPoolBlock4(x)
        # torch.Size([64, 200, 1, 6])
        # 200x6 = 1200
        # reshape to (64, 1200)
        x = x.view(x.shape[0],-1)
        output = self.ClassificationLayer(x)
        return output
