import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from torch.nn import BatchNorm2d as bn



def conv3x3(in_planes, out_planes, stride=1,padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ExitBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rate=2, downsample=None):
        super(ExitBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetNoFc(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetNoFc, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(ExitBlock, 256, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_ds4 = self.layer1(x)
        x_ds8 = self.layer2(x_ds4)
        x_ds16 = self.layer3(x_ds8)
       # print('resnet_finish')
        x_ds32 = self.layer4(x_ds16)
        #print('resnet_finish')
        return x ,x_ds32


def resnet18(**kwargs):
    model = ResNetNoFc(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

class atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(atrous_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                   bn(64),
                                   nn.ReLU())
        self.atrous_6 = nn.Sequential(
            atrous_module(256,64,rate=1),
            atrous_module(64, 64, rate=6))
        self.atrous_12 = nn.Sequential(
            atrous_module(256,64,rate=1),
            atrous_module(64, 64, rate=12))
        self.atrous_18 = nn.Sequential(
            atrous_module(256,64,rate=1),
            atrous_module(64, 64, rate=18))
        self.avg_pool = nn.Sequential(atrous_module(256,64,rate=1),
                                      nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Upsample(scale_factor=32, mode='bilinear'))
        self.conv2 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=1),
                                   bn(128),
                                   nn.ReLU())
    def forward(self, input):
            x_1 = self.conv1(input)
            x_6 = self.atrous_6(input)
            x_12 = self.atrous_12(input)
            x_18 = self.atrous_18(input)
            x_avg = self.avg_pool(input)
            x = torch.cat((x_1, x_6, x_12, x_18, x_avg),dim=1)
            x = self.conv2(x)
            return x



class BR(nn.Module):
    def __init__(self, feature_num):
        super(BR, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return x + self.transform(x)

class Deeplabv3_plus_resnet18(nn.Module):
    def __init__(self,n_classes=6):
        super(Deeplabv3_plus_resnet18, self).__init__()
        self.resnet = resnet18()
        self.encoder = Encoder()
        self.first_conv = nn.Sequential(nn.Conv2d(64, 48, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU())
        #self.br1 = BR(304)
        self.last_conv = nn.Sequential(nn.Conv2d(176, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        #self.br2 = BR(256)
        self.classification = nn.Conv2d(128, n_classes, kernel_size=1, stride=1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self._initialize_weights()

    def forward(self, input, target=None):
            low_feature, high_feature = self.resnet(input)
            high_feature = self.encoder(high_feature)
            low_feature = self.first_conv(low_feature)

            high_feature = self.upsample_4(high_feature)
            feature = torch.cat((low_feature, high_feature),dim=1)
            #feature = self.br1(feature)

            feature = self.last_conv(feature)
            feature = self.upsample_4(feature)
            #feature = self.br2(feature)
            out = self.classification(feature)
            softmax_output = F.softmax(out)

            mask_out = Variable(softmax_output.data.max(1)[1])
            out_forshow = torch.max(softmax_output, dim=1)[1]

            if target is not None:
                pairs = {'out': (out, target),
                         'mask_out': (mask_out, target)
                         }
                return pairs, self.exports(input, out_forshow * np.float(255.0), target * np.float(255.0))
            else:
                return self.exports(input, out_forshow, softmax_output)

    def exports(self, x, output, target):
            result = {'input': x, 'output': output}
            if target is not None:
                result['soft_out'] = target
            return result

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

def netParams(model):
            '''
            helper function to see total network parameters
            :param model: model
            :return: total network parameters
            '''
            total_paramters = 0
            for parameter in model.parameters():
                i = len(parameter.size())
                p = 1
                for j in range(i):
                    p *= parameter.size(j)
                total_paramters += p

            return total_paramters

def get_FileSize(file_path):
            import os
            file_path = unicode(file_path, 'utf8')
            fsize = os.path.getsize(file_path)
            fsize = fsize / float(1024 * 1024)
            return round(fsize, 2)

if __name__ == "__main__":
            import time

            images = Variable(torch.randn(1, 3, 512, 512))
            d = Deeplabv3_plus_resnet18()
            d = nn.DataParallel(d).cuda()
            print (d)
            print ("do forward...")
            start_time = time.time()
            outputs = d(images)
            import os

            print('process ', os.path.basename(__file__))
            print (outputs['soft_out'].size())
            print('time:', time.time() - start_time)
            print('total parameter:', netParams(d))
            model_name = 'resnet_gcn_seg.pth'
            torch.save(d.state_dict(), model_name)
            print('model size:', get_FileSize(model_name), 'MB')


