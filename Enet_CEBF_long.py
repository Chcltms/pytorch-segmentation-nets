


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.setrecursionlimit(100000)


class Bottleneck3x3(nn.Module):
    def __init__(self, inplanes, planes, pad=1, dilation=1):
        super(Bottleneck3x3, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class Bottleneck5x5(nn.Module):
    def __init__(self, inplanes, planes):
        super(Bottleneck5x5, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(planes, planes, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckDown2(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDown2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.convm = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes)
        )

        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual, indices = F.max_pool2d(residual, kernel_size=2, stride=2, return_indices=True)
        residual = self.convm(residual)
        out += residual
        out = self.prelu(out)

        return out, indices


class BottleneckDim_Res(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim_Res, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.PReLU(planes),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.PReLU(planes),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )
        self.resconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes)
        )

        self.prelu = nn.PReLU(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.resconv(residual)
        out += residual

        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

        return out


class BottleneckDim(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.PReLU(planes),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.PReLU(planes),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )

        self.prelu = nn.PReLU(outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

        return out


class BottleneckUp(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(planes, planes, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )

    def forward(self, x):
        out = self.convs(x)
        return out


class BottleneckUp_Res(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp_Res, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(planes, planes, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )


    def forward(self, x):
        out = self.convs(x)

        return out


class BottleneckUp_Res_spacial(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp_Res_spacial, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(planes, planes, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )


    def forward(self, x):
        out = self.convs(x)

        return out

class CBR(nn.Module):
    def __init__(self, inplane, outplane, stride=1):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outplane),
            nn.PReLU(outplane)
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class denseLayer(nn.Module):
    def __init__(self, inplane, midplane, outplane):
        super(denseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inplane),
            nn.PReLU(inplane),
            nn.Conv2d(inplane, midplane, kernel_size=1),
            nn.BatchNorm2d(midplane),
            nn.PReLU(midplane),
            nn.Conv2d(midplane, outplane, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        new_feature = self.conv(x)
        out = torch.cat((x, new_feature), dim=1)
        return out

class denseBlock(nn.Module):
    def __init__(self, num_layer, inplane, midplane):
        super(denseBlock, self).__init__()
        self.num_layer = num_layer
        self.blocks = nn.ModuleList()
        for idx in range(num_layer):
            self.blocks.append(denseLayer(inplane*(idx+1), midplane, inplane))
        self.conv_down = nn.Conv2d(inplane * (num_layer+1), inplane, kernel_size=1)

    def forward(self, x):
        for idx in range(self.num_layer):
            x = self.blocks[idx](x)
        out = self.conv_down(x)
        return out


# fabby ver
class EnetCEB(nn.Module):
    def __init__(self, num_class=2, init_state_path=None):
        super(EnetCEB, self).__init__()

        # init section
        self.init_conv = nn.Conv2d(3, 13, kernel_size=7, stride=4, padding=3)  # video
        self.init_bn = nn.BatchNorm2d(16)
        self.init_prelu = nn.PReLU(16)  # nessary
        self.init_x = denseBlock(3, 16, 16)

        # section 1
        self.bottle1_downDim = 16
        self.bottle1_Dim = 64
        self.bottle1_1 = CBR(16, 16, stride=2)
        self.bottle1_x = denseBlock(5, 16, 16)

        # section 2
        self.bottle2_downDim = 32
        self.bottle2_Dim = 128
        self.bottle2_1 = CBR(16*2, 32, stride=2)
        self.bottle2_x = denseBlock(5, 32, 32)

        self.bottle3_1 = CBR(32*2, 32)

        self.bottle3_bi2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bottle3_bi4 = nn.UpsamplingNearest2d(scale_factor=4)
        self.bottle3_downC1 = nn.Conv2d(32, 64, kernel_size=1)
        self.bottle3_downC2 = nn.Conv2d(32, 16, kernel_size=1)

        # section 4
        self.bottle4_1up = BottleneckUp_Res(32, self.bottle1_downDim, self.bottle1_Dim)
        self.bottle4_2 = BottleneckDim_Res(self.bottle1_Dim, 32, 64, usePrelu=False)
        self.bottle4_3 = BottleneckDim(64, 32, 64, usePrelu=False)
        self.bottle4_downC = nn.Conv2d(16*2, self.bottle1_Dim, kernel_size=1)

        self.bottle4_bi = nn.UpsamplingNearest2d(scale_factor=2)
        self.bottle4_down = nn.Conv2d(64, 16, kernel_size=1)

        # section 5
        self.bottle5_1up = BottleneckUp_Res_spacial(64, 16, 16)
        self.bottle5_2 = BottleneckDim_Res(16, 4, 16, usePrelu=False)
        # self.bottle5_downC = nn.Conv2d(self.bottle1_Dim*2, 16, kernel_size=1)

        # section 6
        self.bottle6_1 = nn.ConvTranspose2d(16, 4, kernel_size=8, padding=2, stride=4)
        self.bottle6_2 = nn.Conv2d(4, num_class, kernel_size=3, padding=1)

        self.weights_init()

        if init_state_path is not None:
            checkpoint = torch.load(init_state_path)
            model_dict = self.state_dict()
            update_dict = {}
            for key in checkpoint:
                if key[7:] in model_dict:
                    update_dict[key[7:]] = checkpoint[key]
            model_dict.update(update_dict)
            self.load_state_dict(model_dict)

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, target=None):
        # init section
        init_out = self.init_conv(x)
        init_mp = F.max_pool2d(x, kernel_size=4, stride=4)
        init_down = torch.cat((init_out, init_mp), 1)
        init_down = self.init_bn(init_down)
        init_down = self.init_prelu(init_down)
        init_down = self.init_x(init_down)

        # section 1
        bottle1_down= self.bottle1_1(init_down)
        bottle1_5 = self.bottle1_x(bottle1_down)
        bottle1_out = torch.cat((bottle1_down, bottle1_5), 1)


        # section 2
        bottle2_down = self.bottle2_1(bottle1_out)
        bottle2_8 = self.bottle2_x(bottle2_down)
        bottle2_out = torch.cat((bottle2_down, bottle2_8), 1)


        # section3
        bottle3_1 = self.bottle3_1(bottle2_out)
        bottle3_bi2 = self.bottle3_bi2(bottle3_1)
        bottle3_bi4 = self.bottle3_bi4(bottle3_1)
        bottle3_downC1 = self.bottle3_downC1(bottle3_bi2)
        bottle3_downC2 = self.bottle3_downC2(bottle3_bi4)



        # section4
        bottle4_1 = self.bottle4_1up(bottle3_1)
        bottle4_downC = self.bottle4_downC(bottle1_out)
        # bottle4_cat = torch.cat((bottle4_1, bottle3_downC1, bottle4_downC), 1)
        bottle4_cat = bottle4_1 + bottle3_downC1 + bottle4_downC
        bottle4_bi = self.bottle4_bi(bottle4_1)
        bottle4_down = self.bottle4_down(bottle4_bi)




        bottle4_2 = self.bottle4_2(bottle4_cat)
        bottle4_3 = self.bottle4_3(bottle4_2)

        # section5
        bottle5_1 = self.bottle5_1up(bottle4_3)
        # bottle5_downC = self.bottle5_downC(bottle5_1)
        # bottle5_cat = torch.cat((bottle5_1, bottle4_down, bottle3_downC2, init_down), 1)
        bottle5_cat = bottle5_1 + bottle4_down + bottle3_downC2 + init_down


        bottle5_2 = self.bottle5_2(bottle5_cat)

        # section6
        bottle6_1 = self.bottle6_1(bottle5_2)
        out = self.bottle6_2(bottle6_1)

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

def get_model_and_input():
    pth_name = "enet_ecb_no_dilation.pth"
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    model = EnetCEB()

    if os.path.isfile(pth_file):
        # model = nn.DataParallel(model).cpu()

        # original saved file with DataParallel
        checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage)

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    else:
        print "Warning :Load pth_file failed. The test result may be inconsistent !!!"

    batch_size = 1
    channels = 3
    height = 224
    width = 224
    images = Variable(torch.ones(batch_size, channels, height, width))
    return model, images

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

    images = Variable(torch.randn(4, 3, 480, 480))
    d = EnetCEB()
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
    model_name = '/root/group-hair-skin-seg/sunbiao/mode_test/enet_encoder_conv_decoder_224.pth'
    torch.save(d.state_dict(), model_name)
    print('model size:', get_FileSize(model_name), 'MB')
