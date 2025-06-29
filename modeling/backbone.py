import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import pickle
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import clip
import random
import pickle

pretrained_weights_path = '/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/all_outs/diverse_weather/model_best.pth'
pretrained_state_dict = torch.load(pretrained_weights_path)
offsets = pretrained_state_dict['model']['offsets'].to('cuda:0')

@BACKBONE_REGISTRY.register()
class ClipRN101(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE 
        self.proj = nn.Linear(512,512)
        self.global_proj = nn.Linear(512,512)
        self.use_proj = cfg.MODEL.USE_PROJ 
        

    def set_backbone_model(self,model):
        self.enc = model
        for name,val in self.enc.named_parameters():
            head = name.split('.')[0]
            if head not in self.unfreeze:
                val.requires_grad = False
            else:
                val.requires_grad = True
        
        self.backbone_unchanged = nn.Sequential(*self.enc.layer3[:19])

    def forward(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        # if random.uniform(0, 1) > 0.4:
        #     y = random.sample(([2,3,4,6,13]),1)
        #     x = x + offsets[y[0]].unsqueeze(0).mean(dim=(2,3),keepdims=True)
        return {"res4": x}


    def forward_l12(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)  
        return x
     
    def forward_l3(self, x):
        
        x = self.enc.layer3(x)
        return {"res4": x}
 
    def output_shape(self):
        return {"res4": ShapeSpec(channels=1024, stride=16)}
    
    def forward_res5(self,x):
        #detectron used last resnet layer for roi heads
        return self.enc.layer4(x)

    def attention_global_pool(self,input):
        x = self.enc.attnpool(input)
        return x
