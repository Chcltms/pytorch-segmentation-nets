import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, num_classes=6, growth_rate=24,
                 reduction=0.5, block=BottleneckBlock, dropRate=0.0,block_config=(6, 12, 24, 16)):
                
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate

        
       # First convolution
        self.features = nn.Sequential(
             nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False),
             nn.BatchNorm2d(in_planes),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 1st block
        self.block1 = DenseBlock(block_config[0], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+block_config[0]*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(block_config[1], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+block_config[1]*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(block_config[2], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+block_config[2]*growth_rate)
        self.trans3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 4rd block
        self.block4 = DenseBlock(block_config[3], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+block_config[3]*growth_rate)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # global average pooling and classifier
        # self.fc = nn.Linear(in_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        tran1 = self.trans1(self.block1(out))
        tran2 = self.trans2(self.block2(tran1))
        tran3 = self.trans3(self.block3(tran2))
        dense4 = self.relu(self.bn1(self.block4(tran3)))
        return dense4,tran3,tran2,tran1


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


class OuputMoudle(nn.Module):
    def __init__(self, feature_num):
        super(OuputMoudle, self).__init__()
        #self.transform0 = nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True)
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
            BR(feature_num),
            nn.ConvTranspose2d(feature_num, feature_num, kernel_size=2, stride=2, padding=0, bias=True),
            
        )

    def forward(self, x):
        #out = self.transform0(x[0])
        out = self.transform1(x[1] + x[0])
        out = self.transform2(x[2] + out)
        out = self.transform3(x[3] + out)

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


class DensenetGCNSegmentation(nn.Module):
    def __init__(self,class_num=6, kernal_size=15, feature_num=21):
        super(DensenetGCNSegmentation, self).__init__()

        self.GCN = nn.Sequential(
            GCNMoudle(features_in_num=[768,384,192,96], feature_out_num=feature_num, kernal_size=kernal_size),
            OuputMoudle(feature_num=feature_num)
        )
        self.DenseNet=DenseNet3()

        self.classification = nn.Conv2d(feature_num, class_num, kernel_size=1, stride=1, padding=0, bias=True)

        self._initialize_weights()

     

    def forward(self, x, target=None):
        desnet_output = self.DenseNet(x)
        gcn_output = self.GCN(desnet_output)
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
    d = DensenetGCNSegmentation()
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
    model_name = '/root/lsb/coco_train/densenet_gcn_seg.pth'
    torch.save(d.state_dict(), model_name)
    print('model size:', get_FileSize(model_name), 'MB')
