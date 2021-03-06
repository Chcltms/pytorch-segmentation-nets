import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np




class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):

    def __init__(self, num_classes=6):

        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.conv1 = nn.Conv2d(3, 32, 3,2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
     

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,512,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        #self.block6=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        #self.block7=Block(512,512,3,1,start_with_relu=True,grow_first=True)

        # self.block8=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        # self.block9=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        # self.block10=Block(512,512,3,1,start_with_relu=True,grow_first=True)
        # self.block11=Block(512,512,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(512,1024,2,2,start_with_relu=True,grow_first=False)

        # self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        # self.bn3 = nn.BatchNorm2d(1536)

       
        # self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        # self.bn4 = nn.BatchNorm2d(2048)

        # self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.relu(x)
        #print(out.shape)
        x_sp1 = self.block1(out)
        #print(x_sp1.shape)
        x_sp2= self.block2(x_sp1)
        #print(x_sp2.shape)
        x_sp3 = self.block3(x_sp2)
        #print(x_sp3.shape)
        x_sp3 = self.block4(x_sp3)   
        x_sp3 = self.block5(x_sp3)
        #x_sp3 = self.block6(x_sp3)
        #x_sp3 = self.block7(x_sp3)
        # x_sp3 = self.block8(x_sp3)
        # x_sp3= self.block9(x_sp3)
        # x_sp3 = self.block10(x_sp3)
        # x_sp3= self.block11(x_sp3)
        x_sp4= self.block12(x_sp3)
        # x_sp4 = self.conv3(x_sp4)
        # x_sp4= self.bn3(x_sp4)
        # x_sp4= self.relu(x_sp4)
        # x_sp4= self.conv4(x_sp4)
        # x_sp4= self.bn4( x_sp4)
        # x_sp4= self.relu( x_sp4)
        #print(x_sp4.shape)
        return x_sp4,x_sp3,x_sp2,x_sp1

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
           
            
        )

    def forward(self, x):
        out = self.transform0(x[0])
        out = self.transform1(x[1] + out)
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


class XceptionGCNSegmentation(nn.Module):
    def __init__(self,class_num=6, kernal_size=15, feature_num=21):
        super(XceptionGCNSegmentation, self).__init__()

        self.GCN = nn.Sequential(
            GCNMoudle(features_in_num=[1024,512,256,128], feature_out_num=feature_num, kernal_size=kernal_size),
            OuputMoudle(feature_num=feature_num)
        )
        self.xception=Xception()

        self.classification = nn.Conv2d(feature_num, class_num, kernel_size=1, stride=1, padding=0, bias=True)

        self._initialize_weights()

     

    def forward(self, x, target=None):
        xception_output = self.xception(x)
        gcn_output = self.GCN(xception_output)
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
	#checkpoint = torch.load('snapshot_72_G_model')
    	#self.load_state_dict(checkpoint)

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

def XceptionGCN_Pretrained_72():
   model = XceptionGCNSegmentation()
   checkpoint = torch.load('snapshot_72_G_model') 
   model_dict = model.state_dict()
   pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
   model_dict.update(pretrained_dict) 
   model.load_state_dict(model_dict)
   return model
    
if __name__ == "__main__":
    import time
   
    images = Variable(torch.randn(1, 3, 512, 512))
    d = XceptionGCNSegmentation()
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
    model_name = '/root/lnx/coco_ADE20K_train/xception_gcn_seg.pth'
    torch.save(d.state_dict(), model_name)
    print('model size:', get_FileSize(model_name), 'MB')


   
 
