# global
from typing import List, Optional, Type, Union
import builtins

# locals
import ivy
import ivy_models


class BasicBlock(ivy.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = ivy.Conv2D(in_planes, planes, [3,3], stride, 1, with_bias=False)
        self.bn1 = ivy.BatchNorm2D(planes)
        self.conv2 = ivy.Conv2D(in_planes, planes, [3,3], stride, 1, with_bias=False)
        self.bn2 = ivy.BatchNorm2D(planes)

        self.shortcut = ivy.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ivy.Sequential(
                ivy.Conv2D(in_planes, self.expansion * planes, [1,1], stride, 1, with_bias=False),
                ivy.BatchNorm2D(self.expansion * planes)
            )

    def _forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

        

class Bottleneck(ivy.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = ivy.Conv2D(in_planes, planes, [1,1], stride, 1, with_bias=False)
        self.bn1 = ivy.BatchNorm2D(planes)
        self.relu = ivy.ReLU()
        
        # self.conv2 = ivy.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = ivy.Conv2D(planes, planes, [3,3], stride, 1, with_bias=False)
        
        self.bn2 = ivy.BatchNorm2D(planes)
        self.conv3 = ivy.Conv2D(planes, self.expansion * planes, [1, 1], stride, 1, with_bias=False)
        self.bn3 = ivy.BatchNorm2D(self.expansion * planes)

        self.shortcut = ivy.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ivy.Sequential(
                ivy.Conv2D(in_planes, self.expansion * planes, [1,1], stride, 1, with_bias=False),
                ivy.BatchNorm2D(self.expansion * planes)
            )

    def _forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        preact = out
        out = self.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class ResNet(ivy.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = ivy.Conv2D(in_channel, 64, [3,3], 1, 1,
                               with_bias=False)
        self.bn1 = ivy.BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))

        for m in self.Modules():
            if isinstance(m, ivy.Conv2D):
                ivy.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (ivy.BatchNorm2D, ivy.GroupNorm)):
                ivy.init.constant_(m.weight, 1)
                ivy.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in ivy.modules():
                if isinstance(m, Bottleneck):
                    ivy.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    ivy.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return ivy.Sequential(*layers)

    def _forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    

class ResNetModel(ivy.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(ResNetModel, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = ivy.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = ivy.Sequential(
                ivy.Linear(dim_in, dim_in),
                ivy.ReLU(inplace=True),
                ivy.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def _forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
