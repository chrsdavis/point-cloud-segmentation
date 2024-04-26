import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.relu3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        y = self.relu2(self.conv2(x))
        y = self.bn1(y)
        y = self.relu3(self.conv3(y))
        y = self.bn2(y)
        return x + y

class BlockI(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.relu3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=2, dilation=2, padding=1)
        self.relu4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(3*out_channels, out_channels, kernel_size=1)
        self.relu5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        y1 = self.bn1(self.relu2(self.conv2(x)))
        y2 = self.bn2(self.relu3(self.conv3(y1)))
        y3 = self.bn3(self.relu4(self.conv4(y2)))
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.bn4(self.relu5(self.conv5(y)))
        return x + y

class BlockII(nn.Module):
    def __init__(self, dropout_rate=0.2, dropout=True):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(p=dropout_rate)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.pool(x)
        return x

class BlockIII(nn.Module):
    def __init__(self, dropout_rate=0.2, dropout=True):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = nn.PixelShuffle(2)(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x

class BlockIV(nn.Module):
    def __init__(self, dropout_rate=0.2, dropout=True):
        super().__init__() 
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, y):
        # y is from the skip connection
        x = torch.cat((x,y), dim=1) 
        if self.dropout:
            x = self.dropout_layer(x)
        return x

class BlockV(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, dropout=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels//4 + 2*out_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=2, dilation=2, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(3*out_channels, out_channels, kernel_size=1)
        self.relu4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(p=dropout_rate)

    def forward(self, x): 
        y1 = self.bn1(self.relu1(self.conv1(x)))
        y2 = self.bn2(self.relu2(self.conv2(y1)))
        y3 = self.bn3(self.relu3(self.conv3(y2)))
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.bn4(self.relu4(self.conv4(y)))

        if self.dropout:
            y = self.dropout_layer(y)

        return y

class SalsaNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.context1 = ContextModule(5, 32)
        self.context2 = ContextModule(32, 32)
        self.context3 = ContextModule(32, 32)

        self.conv_down1 = BlockI(32, 64)
        self.proc_down1 = BlockII(dropout=False)

        self.conv_down2 = BlockI(64, 128)
        self.proc_down2 = BlockII()

        self.conv_down3 = BlockI(128, 256)
        self.proc_down3 = BlockII()

        self.conv_down4 = BlockI(256, 256)
        self.proc_down4 = BlockII()

        self.conv_down5 = BlockI(256, 256)

        self.proc_up1 = BlockIII()
        self.cat_up1 = BlockIV()
        self.conv_up1 = BlockV(256, 128)

        self.proc_up2 = BlockIII()
        self.cat_up2 = BlockIV()
        self.conv_up2 = BlockV(128, 128)

        self.proc_up3 = BlockIII()
        self.cat_up3 = BlockIV()
        self.conv_up3 = BlockV(128, 64)

        self.proc_up4 = BlockIII(dropout=False)
        self.cat_up4 = BlockIV(dropout=False)
        self.conv_up4 = BlockV(64, 32)

        self.last_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.context1(x)
        x = self.context2(x)
        x = self.context3(x)

        x = self.conv_down1(x)
        x1 = x
        x = self.proc_down1(x)

        x = self.conv_down2(x)
        x2 = x
        x = self.proc_down2(x)

        x = self.conv_down3(x)
        x3 = x
        x = self.proc_down3(x)

        x = self.conv_down4(x)
        x4 = x
        x = self.proc_down(x)

        x = self.conv_down5(x)

        x = self.proc_up1(x)
        x = self.cat_up1(x, x4)
        x = self.conv_up1(x)

        x = self.proc_up2(x)
        x = self.cat_up2(x, x3)
        x = self.conv_up2(x)

        x = self.proc_up3(x)
        x = self.cat_up3(x, x2)
        x = self.conv_up3(x)

        x = self.proc_up4(x)
        x = self.cat_up4(x, x1)
        x = self.conv_up4(x)

        x = self.last_conv(x)
        return F.softmax(x, dim=1)

