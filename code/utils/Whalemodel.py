import timm
from .triplet_loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class model_whale(nn.Module):
    def __init__(self, num_classes=15587, inchannels=3, model_name='senet154'):
        super(model_whale,self).__init__()
        planes = 2048
        self.basemodel = timm.create_model('senet154', pretrained=False, in_chans=3)
        local_planes = 512
        self.local_conv = nn.Conv2d(planes, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(planes)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift
        # self.archead = Arcface(embedding_size=planes, classnum=num_classes, s=64.0)
        self.fc = nn.Linear(planes, num_classes)
#         init.normal_(self.fc.weight, std=0.001)
#         init.constant_(self.fc.bias, 0)
    def forward(self, x, label=None):
        x = self.basemodel.conv1(x)
        x = self.basemodel.bn1(x)
        x = self.basemodel.act1(x)
        x = self.basemodel.maxpool(x)
        x = self.basemodel.layer1(x)
        x = self.basemodel.layer2(x)
        x = self.basemodel.layer3(x)
        feat = self.basemodel.layer4(x)
        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        # local feat
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        out = self.fc(global_feat) * 16
        return global_feat, local_feat, out
    
    def getLoss(self, global_feat, local_feat, results,labels):
        triple_loss = global_loss(TripletLoss(margin=0.3), global_feat, labels)[0] + \
                      local_loss(TripletLoss(margin=0.3), local_feat, labels)[0]
        loss_ = sigmoid_loss(results, labels, topk=30)
        self.loss = triple_loss + loss_
        
if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    test_net = model_whale(num_classes=15587, inchannels=3).cuda()
    input = torch.rand((4, 3, 256, 512)).cuda()
    out = test_net(input)
    print(out[0].shape,out[1].shape,out[2].shape)
