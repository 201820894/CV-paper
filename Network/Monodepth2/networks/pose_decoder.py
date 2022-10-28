

from typing import OrderedDict
import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()
        
        self.num_ch_enc=num_ch_enc
        self.num_input_features=num_input_features
        
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for=num_input_features-1
        self.num_frames_to_predict_for=num_frames_to_predict_for
        
        self.convs=OrderedDict()
        #Image size to 256
        self.convs[("squeeze")]= nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        
        self.convs[("pose", 0)]= nn.Conv2d(num_input_features*256, 256, 3, stride, 1)
        self.convs[("pose", 1)]= nn.Conv2d(256, 256, 3, stride, 1)
        # last: 1x1 conv same size
        self.convs[("pose", 2)]= nn.Conv2d(256, 6*num_frames_to_predict_for, 1)
        
        self.relu=nn.ReLU()
        self.net=nn.ModuleList(list(self.convs.values()))
        
    def forward(self, input_features):
        # Output of resnet encoder 
        last_features=input_features[-1]
        # Reduce channel to 256 and adapt relu non-linearity
        cat_features=self.relu(self.convs["squeeze"](last_features)) 
        
        out=cat_features
        for i in range(3):
            out=self.convs[("pose", i)](out)
            if i!=2:
                out=self.relu(out)
        # 결국 목표는 pose matrix를 학습하는 것이기 때문에 맞는 format으로 설정
        # (B, 6*num_frames_to_predict_for, H, W) -> (B, 6*num_frames_to_predict_for)
        out=out.mean(3).mean(2)
        
        # (B, num_frames_to_predict_for, 1, 6) 
        # 0.01은 prediction 자체를 너무 작게하면 변화에 둔감해질 수 있으므로?
        out=0.01*out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        # (B, num_frames_to_predict_for, 1, 3)
        axisangle= out[..., :3]
        # (B, num_frames_to_predict_for, 1, 3)
        translation= out[..., 3:]
        
        # 우선 axisangle 관련 3개, translation 관련 3개의 숫자를 return 한거
        # (B, 2, 1, 3) 을 2개 반환
        return axisangle, translation
        
    