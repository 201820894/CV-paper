from sre_parse import ESCAPES
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# depth = (baseline * focal length) / disparity)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    # Depth decoder 뒤에 이어지는듯, dispnet이랑 같이 한번더보기
    min_disp = 1/max_depth
    max_disp = 1/min_depth
    scaled_disp = min_disp+(max_disp-min_disp)*disp
    depth = 1/scaled_disp

    return scaled_disp, depth


# https://nobilitycat.tistory.com/entry/%EC%9E%84%EC%9D%98%EC%9D%98-%EC%B6%95-%ED%9A%8C%EC%A0%84-Axis-Angle-Rotation
# x, y, z: axis
# C: 1-cos
# ca, sa: cos, sin
# vec: (B, 1, 3)
# num_frames_to_predict = 2
# 의문점은 왜 angle을 axis의 norm으로 정의하는지: 그냥 한번에 예측하기 위해서인지
def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)

    axis = vec/(angle+1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)

    C = 1-ca
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(
        device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T



def transformation_from_parameters(axisangle, translation, invert=False):

    # 찐막 target->source임: 근거: depth를 target거만 구함 
    # 방향이야 뭐 그런가 싶다.
    # -1 ->0 이 잘못된거니까 invert 하는거

    # (B, 4, 4)
    R = rot_from_axisangle(axisangle)
    
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1
    # (B, 4, 4)
    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M


class Conv3x3(nn.Module):
    # pad and convolve input

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)

        return out


class ConvBlock(nn.Module):
    # Layer to perform a convolution followed by ELU

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)

        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image to pointcloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        # indexing='xy' : cartesian
        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy'
        )
        # (2, w, h)
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # 좌표를 nn.parameter로 넘겨주기
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=True)

        #(B, 1, hxw)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height*self.width),
                                 requires_grad=False)

        # 맨 앞에 1차원 생성해서 batch size만큼 repeat 할 수 있게
        # (1, 2, hxw)
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)

        # 각 dimension을 몇번씩 반복할건지
        # (B, 2, hxw)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)

        # (B, 3, hxw)
        # homogeneous 좌표계로 표현
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    """ self.pix_coords
    각 batch 연산에 사용되는 (height 방향 좌표, width 방향 좌표, 1) 경우의 수 생성
        self.pix_coords : 
        - batch_size = 2
        - height = 2
        - width = 3
        
        Parameter containing:
    tensor([[[0., 1., 2., 0., 1., 2.],
            [0., 0., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.]],
            
            [[0., 1., 2., 0., 1., 2.],
            [0., 0., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.]]])
    
    """

    def forward(self, depth, inv_K):

        # homogeneous pixel coordinate -> homogenous normalized coordinate
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)

        # cam point depth 곱하기 -> projective geometry 관점, 다시 원본 point로
        cam_points = depth.view(self.batch_size, 1, -1)*cam_points

        # 다시 homogenous 형태로
        cam_points = torch.cat([cam_points, self.ones], 1)

        # Cam points 는 말 그대로 카메라 좌표계에서 x, y, z homogeneous 형태로 표현한거
        # 카메라 좌표계랑 월드 좌표계랑 동일하게 놓은거: t=0 frame에 대해서
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # Intrinsic.dot(Transfromation matrix)
        # 변환된 이미지의 intrinsic
        P = torch.matmul(K, T)[:, :3, :]

        # pixel_coordinate = (K.dot(T).dot(cam_points))/s
        # (B, 3, h*w)
        cam_points = torch.matmul(P, points)

        # 마지막꺼 떼기
        pix_coords = cam_points[:, :2, :] / \
            (cam_points[:, 2, :].unsqueeze(1)+self.eps)

        # (batch, 2, height, width)
        # Meshgrid 형태
        pix_coords = pix_coords.view(
            self.batch_size, 2, self.height, self.width)

        # (B, h, w, 2)
        # 여기서 2는 h or w (5,6) 이런식으로 담겨있는걸 뜻함
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # pix_coords의 각 X, Y 좌표의 범위를 (-1 ~ 1) 사이의 범위로 리스케일하여 F.grid_sample에 사용함
        # 0~1로 scale
        pix_coords[..., 0] /= self.width-1
        pix_coords[..., 1] /= self.height-1
        # (-1 ~ 1)로 scale
        pix_coords = (pix_coords-0.5)*2
        return pix_coords


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        #
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x)
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)