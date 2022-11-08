import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import torch

def pil_loader(path):
    with open(path, 'rb') as f: # Read binary: PIL image는 binary로 읽어야
        with Image.open(f) as img: # Open image
            #(H, W, 3)
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    # eigen zhou
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, is_train=False, img_ext='.jpg'):
        super(MonoDataset, self).__init__()
        
        self.data_path=data_path
        self.filenames=filenames
        self.height=height
        self.width=width
        self.num_scales=num_scales
        self.interp=Image.ANTIALIAS # High res -> Low res
        
        self.frame_idxs=frame_idxs
        
        self.is_train=is_train
        self.img_ext=img_ext
        
        self.loader=pil_loader
        self.to_tensor=transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        # 되면 얘로
        #try:
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        #transforms.ColorJitter.get_params(
        #    self.brightness, self.contrast, self.saturation, self.hue)
        
        # 안되면 얘로
        #except TypeError: # Different augmentation
        #    self.brightness = 0.2
        #    self.contrast = 0.2
        #    self.saturation = 0.2
        #    self.hue = 0.1
        
        self.resize={}
        for i in range(self.num_scales):
            s=2**i
            # i: 0, 1, 2, 3
            # interpolation: ANTIALIAS
            self.resize[i]=transforms.Resize((self.height//s, self.width//s), interpolation=self.interp)     
        
        # Check wheter depth file is available    
        self.load_depth=self.check_depth()
        
    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # 여기는 color_aug 아님
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # 여기가 color_aug, scale된 애들에 대해서도 color aug, 그냥 return 그대로
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """  
        
        # About File Format
        inputs={}
        do_color_aug=self.is_train and random.random()>0.5
        do_flip=self.is_train and random.random()>0.5

        # 2011_09_26/2011_09_26_drive_0022_sync 473 r 형태
        # self.filenames에서 여러개 filename 인덱스 받아서 split하기
        line=self.filenames[index].split()
        folder=line[0]

        if len(line)==3:
            # 이거에 0이랑 마지막은 제외되어있음
            frame_index=int(line[1])# 각 scene에서 몇 번째 frame인지?
            
            side=line[2]  # right, left
        else:
            frame_index=0
            side=None
            
        for i in self.frame_idxs: #-1, 0, 1
            # Get raw image(can be flipped): Dataset마다 다른 함수
            # 여기서 inputs[("color", i, -1)] 에는 original index에 대한 정보 없음
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
            
        for scale in range(self.num_scales):
            K=self.K.copy
            K[0,:]*=self.width//(2**scale)
            K[1,:]*=self.height//(2**scale)
            inv_K=np.linalg.pinv(K)
            
            inputs[("K", scale)]=torch.from_numpy(K)       
            inputs[("inv_K", scale)]=torch.from_numpy(inv_K)
            
        if do_color_aug:
            color_aug=transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        
        else: 
            color_aug=(lambda x: x)
            
        self.preprocess(inputs, color_aug)
        
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            # If depth file is avalable
        if self.load_depth:
            # get_depth: Dataset마다 다른 함수: Depth image를 return 
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(
                inputs["depth_gt"].astype(np.float32))

        #if "s" in self.frame_idxs:
        #    stereo_T = np.eye(4, dtype=np.float32)
        #    baseline_sign = -1 if do_flip else 1
        #    side_sign = -1 if side == "l" else 1
        #    stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        #
        #    inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

        