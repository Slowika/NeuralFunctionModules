import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, mixup_process
from utils import get_lambda
import random

from models.attentive_densenet import AttentiveDensenet


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_channels, num_classes, per_img_std= False, stride=1, use_attentive_densenet=True, ad_heads=4, att_sparsity=None):
        super(PreActResNet, self).__init__()
        self.num_heads = ad_heads
        self.key_size = 16
        self.val_size = 16
        self.use_attentive_densenet = use_attentive_densenet

        print("TURNED OFF ALL RESIDUAL STYLE SKIP CONNECTIONS")
        print("Using attentive densenet?", use_attentive_densenet)


        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        #import pdb; pdb.set_trace()
        self.use_attentive_densenet = use_attentive_densenet
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1, use_extra_inp=False)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
 
        share_variant = False
        print('sharing parameters between passes?', share_variant)

        self.in_planes = initial_channels
        
        if share_variant:
            self.conv1_b = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.layer1_b = self.layer1
            self.layer2_b = self.layer2
            self.layer3_b = self.layer3
            self.layer4_b = self.layer4
            #self.linear_b = nn.Linear(initial_channels*8*block.expansion, num_classes)

            for lay in [self.layer1_b, self.layer2_b, self.layer3_b, self.layer4_b]:
                for sub_lay in lay:
                    sub_lay.bn1 = nn.BatchNorm2d(sub_lay.bn1.num_features)
                    sub_lay.bn2 = nn.BatchNorm2d(sub_lay.bn2.num_features)

                    if hasattr(sub_lay, 'shortcut'):
                        sub_lay.shortcut[0] = nn.Conv2d(in_channels=sub_lay.shortcut[0].in_channels, out_channels=sub_lay.shortcut[0].out_channels, kernel_size=1, stride=sub_lay.shortcut[0].stride, bias=False)
                        print('remaking shortcut - non-shared shortcut version')

        else:
            self.conv1_b = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.layer1_b = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
            self.layer2_b = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
            self.layer3_b = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
            self.layer4_b = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
            #self.linear_b = nn.Linear(initial_channels*8*block.expansion, num_classes)

        #self.linear = nn.Linear(initial_channels*8*block.expansion, num_classes)

        #layer_channels = [initial_channels] + num_blocks[0]*[initial_channels] + 
        mult = 1
        layer_channels = [initial_channels*2 * mult] + [initial_channels*4 * mult] + [initial_channels*8 * mult] + [initial_channels * mult] + [initial_channels*2 * mult] + [initial_channels*4 * mult] + [initial_channels*8 * mult]

        if self.use_attentive_densenet:
            self.ad = AttentiveDensenet(layer_channels, self.key_size, self.val_size, self.num_heads, att_sparsity)
            print("Layer channels", layer_channels)

    def _make_layer(self, block, planes, num_blocks, stride, use_extra_inp=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def compute_h1(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        
        all_layers = []
        if self.use_attentive_densenet:
            self.ad.reset()

        if mixup_hidden:
            layer_mix = random.randint(0,2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        
        out = x
        
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)
        
        if target is not None :
            target_reweighted = to_one_hot(target,self.num_classes)
        
        if layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        x_mix = out*1.0

        out = self.conv1(out)

        for sub_layer in self.layer1:
            out = sub_layer(out)

        for sub_layer in self.layer2:
            out = sub_layer(out)

        if self.use_attentive_densenet:
            out = self.ad(out, read=False, write=True)

        for sub_layer in self.layer3:
            out = sub_layer(out)
        
        if self.use_attentive_densenet:
            out = self.ad(out, read=False, write=True)

        for sub_layer in self.layer4:
            out = sub_layer(out)
        
        if self.use_attentive_densenet:
            out = self.ad(out, read=False, write=True)

        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        #out_first = self.linear(out)

        if self.use_attentive_densenet:
            out = x_mix
            out = self.conv1_b(out)
            out = self.layer1_b(out)
            out = self.ad(out, read=True, write=True)
            out = self.layer2_b(out)
            out = self.ad(out, read=True, write=True)
            out = self.layer3_b(out)
            out = self.ad(out, read=True, write=True)
            out = self.layer4_b(out)
            out = self.ad(out, read=True, write=True)
            out = F.interpolate(out, [5, 5])


            #out = F.avg_pool2d(out, 4)

            #out = out.view(out.size(0), -1) #!!!
            #out = self.linear_b(out) #!!!
        else:
            out = F.interpolate(out, [5, 5])

        if target is not None:
            return out, target_reweighted
        else: 
            return out

def preactresnet18(initial_channels, num_classes=10, dropout = False,  per_img_std = False, stride=1, use_ad=True, ad_heads=4, att_sparsity=None):
    return PreActResNet(PreActBlock, [2,2,2,2], initial_channels, num_classes,  per_img_std, stride= stride, use_attentive_densenet=use_ad, ad_heads=ad_heads,att_sparsity=att_sparsity)

def preactresnet34(initial_channels, num_classes=10, dropout = False,  per_img_std = False, stride=1,use_ad=False, ad_heads=4):
    return PreActResNet(PreActBlock, [3,4,6,3], initial_channels, num_classes,  per_img_std, stride= stride, use_attentive_densenet=use_ad, ad_heads=ad_heads)

def preactresnet50(initial_channels, num_classes=10, dropout = False,  per_img_std = False, stride=1,use_ad=False, ad_heads=4):
    return PreActResNet(PreActBottleneck, [3,4,6,3], initial_channels, num_classes,  per_img_std, stride= stride, use_attentive_densenet=use_ad, ad_heads=ad_heads)

def preactresnet101(initial_channels, num_classes=10, dropout = False,  per_img_std = False, stride=1,use_ad=False, ad_heads=4):
    return PreActResNet(PreActBottleneck, [3,4,23,3], initial_channels, num_classes, per_img_std, stride= stride, use_attentive_densenet=use_ad, ad_heads=ad_heads)

def preactresnet152(initial_channels, num_classes=10, dropout = False,  per_img_std = False, stride=1,use_ad=False, ad_heads=4):
    return PreActResNet(PreActBottleneck, [3,8,36,3], initial_channels, num_classes, per_img_std, stride= stride, use_attentive_densenet=use_ad, ad_heads=ad_heads)

