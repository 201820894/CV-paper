import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        # block, layers는 parent class에 전달할 매개변수
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplnes=64
        
        # 왜 num_input_image를 넣는지? 아 conv 어차피 채널 수니까~
        self.conv1=nn.Conv2d(
            num_input_images*3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(block, 64, layers[0])
        self.layer1=self._make_layer(block, 128, layers[1], stride=2)
        self.layer1=self._make_layer(block, 256, layers[2], stride=2)
        self.layer1=self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        
def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks={18:[2,2,2,2], 50:[3,4,6,3]}[num_layers]
    block_type={18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]  
    model=ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    
    if pretrained:
        loaded=model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        # 1차원으로 이미지 수만큼 합치기 + normalize: dim 번째 차원이 늘어나는거
        loaded['conv1.weight']=torch.cat(
            [loaded['conv1.weight']]*num_input_images, 1)/num_input_images
        
        model.load_state_dict(loaded)
        return model
    
class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        
        self.num_ch_enc=np.array([64, 64, 128, 256, 512])
        
        resnets={18: models.resnet18,
                 34: models.resnet34,
                 50: models.resnet50,
                 101: models.resnet101,
                 152: models.resnet152}
        
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        
        #return model
        if num_input_images>1:
            self.encoder=resnet_multiimage_input(num_layers, pretrained, num_input_images)        
        else:
            self.encoder=resnets[num_layers](pretrained)
        
        if num_layers>34:
            self.num_ch_enc[1:]*=4
    
        # List에 각 layer output이 들어감
    def forward(self, input_image):
        self.features=[]
        x=(input_image-0.45)/0.225
        x=self.encoder.conv1(x)
        x=self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        
        # List 전체를 return
        return self.features
            