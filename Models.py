# -*- conding: UTF-8 -*-


import torch
import torch.nn as nn
from collections import OrderedDict


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=9, kernel_size=3, padding=1):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", kernel_size=kernel_size, padding=padding)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.encoder2 = UNet._block(features, features * 2, name="enc2", kernel_size=kernel_size, padding=padding)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", kernel_size=kernel_size, padding=padding)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck", kernel_size=kernel_size, padding=padding)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(1,2), stride=(1,2))
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", kernel_size=kernel_size, padding=padding)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(1,2), stride=(1,2))
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", kernel_size=kernel_size, padding=padding)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(1,2), stride=(1,2))
        self.decoder1 = UNet._block(features * 2, features, name="dec1", kernel_size=kernel_size, padding=padding)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = torch.relu(self.conv(dec1)).squeeze()

        return output

    @staticmethod
    def _block(in_channels, features, kernel_size, padding, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=kernel_size,padding=padding,bias=False,),),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(0.2,inplace=True)),
                    (name + "conv2",nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,bias=False,),),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(0.2,inplace=True)),
                ]
            )
        )


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_num = 14
        
        self.main = nn.Sequential(
            nn.Conv1d(1, self.feat_num*2, 3, 2),
#             nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(self.feat_num*2,self.feat_num*4,3),
#             nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.AvgPool1d(2, stride=2)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2,inplace=True)
        
        self.conv3 = nn.Conv1d(self.feat_num*4,self.feat_num*4,10)
        self.fc1 = nn.Linear(self.feat_num*4,self.feat_num*4)
        self.fc2 = nn.Linear(self.feat_num*4,96)
        self.dp = nn.Dropout(0.5)
            
    def forward(self, trace):
        
        x = self.main(trace.unsqueeze(1))

        c = self.conv3(x)
        c = self.lrelu(c)
        
        c = c.squeeze(-1)

        c = self.fc1(c)
        c = self.lrelu(c)
        c = self.dp(c)
        c = self.fc2(c)
        c = self.relu(c)
        
        return c

		
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_num = 32
        self.init_len = 96
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2,inplace=True)
        
        self.fc1 = nn.Linear(self.init_len,self.feat_num*2)
        self.fc2 = nn.Linear(self.feat_num*2,self.feat_num*4)
        self.fc3 = nn.Linear(self.feat_num*4,self.feat_num*4)
        self.fc4 = nn.Linear(self.feat_num*4,self.feat_num*2)
        self.fc5 = nn.Linear(self.feat_num*2,self.init_len)
        self.dp = nn.Dropout(0.5)

    def forward(self, trace):
        
        c = self.fc1(trace)
        c = self.lrelu(c)

        c = self.fc2(c)
        c = self.lrelu(c)
        
        c = self.fc3(c)
        c = self.lrelu(c)
        
        c = self.dp(c)
        c = self.fc4(c)
        c = self.lrelu(c)
        
        c = self.dp(c)
        c = self.fc5(c)
        c = self.relu(c)
        
        return c


