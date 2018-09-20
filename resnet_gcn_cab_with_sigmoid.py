import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from torch.nn import BatchNorm2d as bn

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


# model_urls = {
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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

    def __init__(self, block, layers, num_classes=6):
        self.inplanes = 64
        super(ResNetNoFc, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x_ds32 = self.layer4(x_ds16)

        return x_ds32, x_ds16, x_ds8, x_ds4


def resnet18(init_state_path=None, **kwargs):
    model = ResNetNoFc(BasicBlock, [2, 2, 2, 2], **kwargs)
    if init_state_path is not None:
        checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    return model


def resnet34(init_state_path=None, **kwargs):
    model = ResNetNoFc(BasicBlock, [3, 4, 6, 3], **kwargs)
    if init_state_path is not None:
        checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    return model


def resnet50(init_state_path=None, **kwargs):
    model = ResNetNoFc(Bottleneck, [3, 4, 6, 3], **kwargs)
    if init_state_path is not None:
        checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    return model


def resnet101(init_state_path=None, **kwargs):
    model = ResNetNoFc(Bottleneck, [3, 4, 23, 3], **kwargs)
    if init_state_path is not None:
        checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    return model


def resnet152(init_state_path=None, **kwargs):
    model = ResNetNoFc(Bottleneck, [3, 8, 36, 3], **kwargs)
    if init_state_path is not None:
        checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
    return model


class GCN(nn.Module):
    def __init__(self, feature_in_num, feature_out_num=21, kernal_size=15):
        super(GCN, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(feature_in_num, feature_out_num, kernel_size=(kernal_size, 1), stride=1,
                      padding=((kernal_size - 1) / 2, 0), bias=True),
            nn.Conv2d(feature_out_num, feature_out_num, kernel_size=(1, kernal_size), stride=1,
                      padding=(0, (kernal_size - 1) / 2), bias=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(feature_in_num, feature_out_num, kernel_size=(1, kernal_size), stride=1,
                      padding=(0, (kernal_size - 1) / 2), bias=True),
            nn.Conv2d(feature_out_num, feature_out_num, kernel_size=(kernal_size, 1), stride=1,
                      padding=((kernal_size - 1) / 2, 0), bias=True)
        )

    def forward(self, x):
        return self.left(x) + self.right(x)


class BR(nn.Module):
    def __init__(self, feature_num=21):
        super(BR, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return x + self.transform(x)


class GCNMoudle(nn.Module):
    def __init__(self, features_in_num, feature_out_num=21, kernal_size=15):
        super(GCNMoudle, self).__init__()
        self.kernal_size = kernal_size
        self.transform = []
        for i in range(len(features_in_num)):
            self.transform.append(self._feature_transform(features_in_num[i], feature_out_num))

        self.transform = nn.ModuleList(self.transform)

    def _feature_transform(self, feature_in_num, feature_out_num):
        transform = nn.Sequential(
            GCN(feature_in_num, feature_out_num, kernal_size=self.kernal_size),
            BR(feature_out_num)
        )

        return transform

    def forward(self, x):
        output = []
        for i in range(len(x)):
            output.append(self.transform[i](x[i]))

        return output

class CAB(nn.Module):
    def __init__(self):
        super(CAB, self).__init__()
        self.score_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(42, 42, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(42, 21, kernel_size=1)

        )
    def forward(self, x_high, x_low):
        x = torch.cat((x_high, x_low), dim=1)
        x = self.score_map(x)
        x = F.sigmoid(x)
        x = x * x_low
        x = x + x_high
        return x

class OuputMoudle(nn.Module):
    def __init__(self, feature_num):
        super(OuputMoudle, self).__init__()
        self.transform0 = nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True)
        self.transform1 = nn.Sequential(
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True)
        )
        self.transform2 = nn.Sequential(
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True)
        )
        self.transform3 = nn.Sequential(
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True),
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True),
            BR(feature_num)
        )
        self.cab1 = CAB()
        self.cab2 = CAB()
        self.cab3 = CAB()

    def forward(self, x):
        out = self.transform0(x[0])
        out = self.cab1(out, x[1])
        out = self.transform1(out)
        out = self.cab2(out, x[2])
        out = self.transform2(out)
        out = self.cab3(out, x[3])
        out = self.transform3(out)

        return out


class MergeModule(nn.Module):
    def __init__(self, feature_num=21, num_level=4):
        super(MergeModule, self).__init__()
        self.transform = []
        for i in range(num_level):
            self.transform.append(self._feature_transform(feature_num))

        self.transform = nn.ModuleList(self.transform)

    def _feature_transform(self, feature_num):
        transform = nn.Sequential(
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True)
        )

        return transform

    def forward(self, x):
        output = []
        output.append(x[0])

        for i in range(1, len(x)):
            output.append(x[i] + self.transform[i - 1](output[i - 1]))

        return output


class UpsampleModule(nn.Module):
    def __init__(self, feature_num=21, upsample_factors=[8, 4, 2, 1]):
        super(UpsampleModule, self).__init__()
        self.transform = []
        for i in range(len(upsample_factors)):
            self.transform.append(self._feature_transform(feature_num, upsample_factors[i]))

        self.transform = nn.ModuleList(self.transform)

    def _feature_transform(self, feature_num, upsample_factor):
        transform = nn.Sequential(
            BR(feature_num),
            nn.Upsample(scale_factor=upsample_factor, mode='bilinear')
            # nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=upsample_factor, padding=0, bias=True)
        )

        return transform

    def forward(self, x):
        output = []
        for i in range(len(x)):
            output.append(self.transform[i](x[i]))

        return output


class ResnetGCN_CAB(nn.Module):
    def __init__(self, class_num=6, resnet_depth=18, kernal_size=15, feature_num=21, init_resnet_state_path=None,
                 init_state_path=None):
        super(ResnetGCN_CAB, self).__init__()

        self.GCN = nn.Sequential(
            #GCNMoudle(features_in_num=[2048, 1024, 512, 256], feature_out_num=feature_num, kernal_size=kernal_size),
            GCNMoudle(features_in_num=[512, 256, 128, 64], feature_out_num=feature_num, kernal_size=kernal_size),
            OuputMoudle(feature_num=feature_num)
        )

        self.classification = nn.Conv2d(feature_num, class_num, kernel_size=1, stride=1, padding=0, bias=True)

        self._initialize_weights()

        if resnet_depth == 18:
            self.resnet = resnet18(init_state_path=init_resnet_state_path)



    def forward(self, x, target=None):
        resnet_output = self.resnet(x)
        gcn_output = self.GCN(resnet_output)
        out = self.classification(gcn_output)

        softmax_output = F.softmax(out)

        mask_out = Variable(softmax_output.data.max(1)[1])
        out_forshow = torch.max(softmax_output, dim=1)[1]

        if target is not None:
            pairs = {'out': (out, target),
                     'out2': (out, target),
                     'mask_out': (mask_out, target)
                     }
            return pairs, self.exports(x, out_forshow * np.float(255.0), target * np.float(255.0))
        else:
            return self.exports(x, out_forshow, softmax_output)

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
    fsize = fsize / float(1024*1024)
    return round(fsize, 2)

if __name__ == "__main__":
    import time

    images = Variable(torch.randn(1, 3, 512, 512))
    d = ResnetGCN_CAB()
    d = nn.DataParallel(d).cuda()
    print (d)
    print ("do forward...")
    start_time = time.time()
    outputs = d(images)
    import os

    print('process ', os.path.basename(__file__))
    print (outputs['output'].size())
    print('time:', time.time() - start_time)
    print('total parameter:', netParams(d))
    model_name = 'resnet_gcn_seg.pth'
    torch.save(d.state_dict(), model_name)
    print('model size:', get_FileSize(model_name), 'MB')
