import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from OT_module.PInet.parameters import Parameters

p = Parameters()

def backward_hook(self, grad_input, grad_output):
    print('grad_input norm:', grad_input[0].data.norm())

def cross_entropy2d(inputs, target, weight=None, size_average=True):
    loss = torch.nn.CrossEntropyLoss()

    n, c, h, w = inputs.size()
    prediction = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    gt =target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    return loss(prediction, gt)

class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, sigmoid=False):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), int(k_size), 
                                                    padding=padding, stride=stride, bias=bias),
                                    nn.BatchNorm2d(int(n_filters)),
                                    nn.ReLU(inplace=True),)
        else:
            if sigmoid:
                self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
                                            nn.Sigmoid(),)
            else:
                self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias)
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

# class Conv2D_BatchNorm_sigmoid(nn.Module):
#     def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True):
#         super(Conv2D_BatchNorm_sigmoid, self).__init__()
#         self.acti=acti
#         if acti:
#             self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), int(k_size), 
#                                                     padding=padding, stride=stride, bias=bias),
#                                     nn.BatchNorm2d(int(n_filters)),
#                                     nn.ReLU(inplace=True),)
#         else:
#             self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
#                                     nn.Sigmoid(),)
#             # self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias)
#             # self.Softmax=nn.Softmax()

#     def forward(self, inputs):
#         outputs = self.cbr_unit(inputs)
#         if not self.acti:
#             print("outputs",outputs.shape)
#             print("outputs_linear",outputs.shape)
#         return outputs


class bottleneck_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_up, self).__init__()
        temp_channels = int(in_channels/4)
        if in_channels < 4:
            temp_channels = int(in_channels)
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels,1,  0, 1)
        self.conv2 = nn.Sequential( nn.ConvTranspose2d(temp_channels, temp_channels, 3, 2, 1, 1),
                                        nn.BatchNorm2d(temp_channels),
                                        nn.ReLU() )
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        return out

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True, sigmoid=False):
        super(bottleneck, self).__init__()
        self.acti = acti
        temp_channels = int(in_channels/4)
        if in_channels < 4:
            temp_channels = int(in_channels)
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1, acti = self.acti, sigmoid=sigmoid)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        return out


class Output(nn.Module):
    def __init__(self, in_size, out_size, sigmoid=False):
        super(Output, self).__init__()
        self.conv = bottleneck(in_size, out_size, acti=False, sigmoid=sigmoid)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class Resnet(nn.Module):
    def __init__(self , model, in_channels, out_channels,num_classes=25, acti = True):
        super(Resnet, self).__init__()
        #取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        self.re1 = bottleneck(int(in_channels), int(out_channels))

        self.up1 = bottleneck_up(int(out_channels),int(out_channels))
        self.up2 = bottleneck_up(int(out_channels),int(out_channels))

        self.out_confidence = Output(int(out_channels), 1, sigmoid=True)      
        self.out_offset = Output(int(out_channels), 2)      
        self.out_instance = Output(int(out_channels), p.feature_size)  

        
    def forward(self, x):
        # print(x.shape)
        # print(self.resnet_layer)
        outputs = self.resnet_layer(x)

        outputs = self.re1(outputs)
        outputs = self.up1(outputs)
        outputs = self.up2(outputs)
        # print(outputs.shape)
        # outputs = self.up3(outputs)
        # print(outputs.shape)
        # print("outputs",outputs.shape)
        # print("out_confidence",self.out_confidence)
        out_confidence = self.out_confidence(outputs)
        out_offset = self.out_offset(outputs)
        out_instance = self.out_instance(outputs)
        
        return [out_confidence, out_offset, out_instance]

if __name__=='__main__':    
    print(model)
