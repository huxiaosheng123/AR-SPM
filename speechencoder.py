import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class SpectrogramNet(nn.Module):
    def __init__(self):
        super(SpectrogramNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2), padding=0, dilation=1, ceil_mode=False),#the best performance among three maxpools
            #nn.MaxPool2d(kernel_size=(2, 6), stride=(2, 4), padding=0, dilation=1, ceil_mode=False),
            #nn.MaxPool2d(kernel_size=(3, 6), stride=(2, 3), padding=0, dilation=1, ceil_mode=False),
        )
        self.CBAM = CBAM_Module(channels=512, reduction=4)
        self.fc1 =  nn.Linear(512 * 4, 4096)
        self.apool = nn.AdaptiveAvgPool2d(1)
        # self.act= nn.Sequential(
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU()
        # )

        #self.fc2 = nn.Linear(4096, 4096)

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        )
        # self.fc3 = nn.Sequential(
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 4096)
        # )

    def forward(self, audio, fuse=False, mean=None):  # 257,598
        #print(audio.shape)
        conv = self.CBAM(self.conv(audio))
        #print(conv.size())
        conv = conv.view(conv.size(0), 1, conv.size(3), -1)
        #print(conv.size())
        fc1 = self.fc1(conv)
        #print(fc1.size())
        fc1 = fc1.view(fc1.size(0), 4096, 1, -1)
        averpool = self.apool(fc1)
        audio_feat = averpool.view(averpool.size(0), -1) # add maen

        #audio_feat = self.act(audio_feat)
        #print(type(mean))
        if fuse == True:
            # print(type(audio_feat))
            # print(mean)
            result = self.fc2(audio_feat+mean)

        else:
            result = self.fc2(audio_feat)
        result1 = F.normalize(result,p=2,dim=1)
        return result1


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
        )
        # self.CBAM = CBAM_Module(channels=64, reduction=4)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2560, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, audio):
        conv = self.conv(audio)
        # print(conv.size())
        feature = conv.view(conv.size(0), -1)
        #print(feature.size())
        result = self.gender_classifier(feature)
        return result

# x = torch.empty(1,2,257,598)
# net = SpectrogramNet()
# output = net(x,False,None)
# print(output.shape)