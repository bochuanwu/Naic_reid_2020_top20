from efficientnet_pytorch import EfficientNet
from .backbones.cls_hrnet import get_cls_net
import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.resnet_ibn_b import resnet101_ibn_b
from .backbones.resnext_ibn_a import resnext101_ibn_a
import torch.nn.functional as F
from torch.nn.parameter import Parameter
class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = model_name
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride)
            
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_b':
            self.in_planes = 2048
            self.base = resnet101_ibn_b(last_stride)
            print('using resnet101_ibn_b as a backbone')
        elif model_name == 'efficientnet_b4':
            print('using efficientnet_b2 as a backbone') 
            self.base = EfficientNet.from_pretrained('efficientnet-b2', advprop=False) 
            self.in_planes = self.base._fc.in_features
        elif model_name == 'efficientnet_b7':
            print('using efficientnet_b7 as a backbone')
            self.base = EfficientNet.from_pretrained('efficientnet-b7', advprop=True)
            self.in_planes = self.base._fc.in_features
        elif model_name == 'resnext101_ibn_a':
            self.in_planes = 2048
            self.base = resnext101_ibn_a(last_stride)
            print('using resnext101_ibn_a as a backbone')
            
        elif model_name == 'HRnet':
            self.in_planes = 2048
            self.base = get_cls_net(cfg, pretrained = model_path)  
            
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet' and model_name != 'efficientnet_b4' and model_name != 'efficientnet_b7' and  model_name != 'HRnet':
       
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if self.model_name == 'HRnet':
            self.attention_tconv = nn.Conv1d(self.in_planes, 1, 3, padding=1)                        
            self.upsample0 = nn.Sequential(
                                    nn.Conv2d(32, self.in_planes, kernel_size=1, stride=1, bias=False),
                                        )
            self.upsample1 = nn.Sequential(
                                    nn.Conv2d(64, self.in_planes, kernel_size=1, stride=1, bias=False),
                                        )
            self.upsample2 = nn.Sequential(
                                    nn.Conv2d(128, self.in_planes, kernel_size=1, stride=1, bias=False),
                                        )
            self.upsample3 = nn.Sequential(
                                    nn.Conv2d(256, self.in_planes, kernel_size=1, stride=1, bias=False),
                                        )
            
    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        if self.model_name == 'HRnet':
            y_list = self.base(x)
            
            global_feat0 = self.gap(self.upsample0(y_list[0])) 
            global_feat1 = self.gap(self.upsample1(y_list[1])) 
            global_feat2 = self.gap(self.upsample2(y_list[2])) 
            global_feat3 = self.gap(self.upsample3(y_list[3])) 
            weight_ori = torch.cat([global_feat0, global_feat1, global_feat2, global_feat3], dim=2)
            weight_ori = weight_ori.view(weight_ori.shape[0], weight_ori.shape[1], -1)
            attention_feat = F.relu(self.attention_tconv(weight_ori))
            attention_feat = torch.squeeze(attention_feat)
            weight = F.sigmoid(attention_feat)
            weight = F.normalize(weight, p=1, dim=1)

            weight = torch.unsqueeze(weight, 1)
            weight = weight.expand_as(weight_ori)
            global_feat = torch.mul(weight_ori, weight)
            global_feat = global_feat.sum(-1)
            global_feat = global_feat.view(global_feat.shape[0], -1) #flatten to (bs, 2048)
            feat = self.bottleneck(global_feat)
            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                return feat
        else:
            if self.model_name =='efficientnet_b4' or self.model_name =='efficientnet_b7':
                x = self.base.extract_features(x)
            else:
                x = self.base(x)
            global_feat = self.gap(x)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)

            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    #print(feat,label)
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)

                return cls_score, global_feat
            else:
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    return feat
                else:
                    # print("Test with feature before BN")
                    return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in param_dict.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        '''
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
            #self.state_dict()[i].copy_(new_state_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
