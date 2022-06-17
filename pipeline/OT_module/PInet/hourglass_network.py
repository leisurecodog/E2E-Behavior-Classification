#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from OT_module.PInet.util_resnet import *

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()
        resnet = models.resnet50(pretrained=False)

        # self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = Resnet(resnet,2048, 128)
        # self.layer2 = hourglass_block(128, 128)


    def forward(self, inputs):
        #feature extraction
        # out = self.resizing(inputs)
        result1 = self.layer1(inputs)     

        return [result1]
