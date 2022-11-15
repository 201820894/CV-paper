from pickle import TRUE
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tensorboardX import SummaryWriter
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
import torchvision


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "height must be a multiple of 32"
        assert self.opt.width % 32 == 0, "width mus be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        # no_cuda가 true 일 때만 cpu이므로 설정 안해주면 자동으로 gpu로 들어감
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # [0, 1, 2, 3]
        # 1/(2**0), 1/(2**1), 1/(2**2) 1/(2**3) 까지의 scale을 가지도록 정의
        # self.num_scales = 4
        self.num_scales = len(self.opt.scales)

        # [0, -1, 1]
        # 각각 [t, t-1, t+1] 번째 frame
        # self.num_input_frames=3
        self.num_input_frames = len(self.opt.frame_ids)
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # pose network에 들어갈 input의 개수
        # (t-1, t), (t, t+1) 처럼 2개의 frame으로 이루어진 쌍을 사용하므로 2로 정의
        # if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.num_pose_frames = 2

        # 스테레오 카메라가 아니므로
        self.use_pose_net = True

        # Depth encoder
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained"
        )
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth decoder
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # Pose encoder
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            # num_input_images=self.num_pose_frames: 2개의 image를 받음
            num_input_images=self.num_pose_frames  # 2
        )
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(
            self.models["pose_encoder"].parameters())

        # Pose decoder
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc, num_input_features=1,
            num_frames_to_predict_for=2
        )
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        """ 여기서 궁금한거: 
        Pose decoder에서 num_input_features 와 num_frames_to_predict_for 의 정확한 의미
        num_frames_to_predict_for 은 예측할 프레임의 수: 2개?: 2개 이미지 넣어서 1개 예측하는거 아니었나?
        -> axisangle이랑 translation 2개씩 return
        num_input_features은 그냥 디코더에 들어가는 input 수
        """

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )

        if self.opt.load_weights_folder is not None:
            # ===================================================================== 마지막 다시보기
            self.load_model()

        print("Training model named:\n ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n ",
              self.opt.log_dir)
        print("Training is using:\n ", self.device)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset, }

        # self.dataset = datasets.KITTIRAWDataset
        self.dataset = datasets_dict[self.opt.dataset]

        # os.path.dirname(__file__)은 해당 파일(trainer.py) 의 디렉토리 반환
        # 이런식으로 쓰는거 생각 익히기
        fpath = os.path.join(os.path.dirname(__file__),
                             "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filename = readlines(fpath.format("val"))
        img_ext = '.png' #if self.opt.png else '.jpg'
        # 해당 파일 열어서 example 수 세기
        num_train_samples = len(train_filenames)
        self.num_total_steps = (
            num_train_samples//self.opt.batch_size) * self.opt.num_epochs

        # KITTIRAWDataset의 객체
        # 각 데이터마다 getitem으로
        # color, color_aug, scale return
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext
        )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filename, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True
        )
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        # Tensorboard
        for mode in ["triain", "val"]:
            self.writers[mode] = SummaryWriter(
                os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        #scales: [0, 1, 2, 3]
        for scale in self.opt.scales:
            h = self.opt.height//(2**scale)
            w = self.opt.width//(2**scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w
            )
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch+1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        # loader에서 가져오는거면 (B, C, H, W)
        # 각 input에 key value 있고 얘들이 batch
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time()-before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            # 일정 시간마다 gt와 비교
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    # 얘가 받는 input은 batch
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        # inputs 구성이 tuple (feature, value) 형식
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # inputs 중에서 frame_id=0, scale=0인 이미지만 Depth Encoder에 입력으로 넣는다.
        # ("color_aug", <frame_id>, <scale>)
        # frame id: -1, 0, 1/ scale은 0가 원상태
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        # depth map
        # 여기서 처음 만들어지고 추가되는거
        # self.outputs[("disp", i)]=self.sigmoid(self.convs[("dispconv", i)](x))
        # i는 scale
        outputs = self.models["depth"](features)

        # pose 예측 결과를 outputs (dict)에 추가함
        # outputs += ("axisangle", 0, f_i), ("translation", 0, f_i), ("cam_T_cam", 0, f_i)
        # 한 batch에 대한 input 전체 넘겨주는거
        outputs.update(self.predict_poses(inputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    # 얘를 바꿔야함
    # batch 단위

    def essential_to_pose(self, inputs):

        outputs = {}
        # scale: 0, frame_id: -1, 0, 1 인거 넣기
        pose_feats = {f_i: inputs["color_aug", f_i, 0]
                      for f_i in self.opt.frame_ids}
        to_grayscale = torchvision.transforms.Grayscale(num_output_channels=1)

        pose_feats_gray = {f_i: torch.squeeze(to_grayscale(pose_feats[f_i]))
                           for f_i in self.opt.frame_ids}

        # 여기까지가 grayscale
        # Resize등, superglue에서 사용하는 전처리 작업 필요

        # -1, 0 또는 0, 1 순서대로 정렬
        for f_i in self.opt.frame_ids[1:]:  # -1, 0
            # (B, 1, H, W) 2방씩
            if f_i < 0:  # -1
                pose_inputs = [pose_feats_gray[f_i], pose_feats_gray[0]]
            else:  # 1
                pose_inputs = [pose_feats_gray[0], pose_feats_gray[f_i]]

            match_points(pose_inputs)

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # scale: 0, frame_id: -1, 0, 1 인거 넣기
        pose_feats = {f_i: inputs["color_aug", f_i, 0]
                      for f_i in self.opt.frame_ids}
        """
        pose feats={0: inputs["color_aug", 0, 0],
                    -1: inputs["color_aug", -1, 0],
                    1: inputs["color_aug", 1, 0],
                    }
                    : Single training item 에 관한거: mono_dataset.py
        """

        # -1, 0 또는 0, 1 순서대로 정렬
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:  # -1
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:  # 1
                pose_inputs = [pose_feats[0], pose_feats[f_i]]
        # Input (B, 3, H, W)
        # (B, 6, H, W)를 넘겨줌
        pose_inputs = self.models["pose_encoder"](torch.cat(pose_inputs, 1))

        # num_input_features= 1
        # num_frames_to_predict_for= 2
        # axisangle : (B, 2, 1, 3)
        # translation : (B, 2, 1, 3)
        axisangle, translation = self.models["pose"](pose_inputs)

        # 0 -> f_i(-1 또는 +1)로의 변화된 카메라 pose의 axisangle, translation을 예측함
        # 여기 하나에 (B, ...) 형식으로 batch 통째로 들어가는거
        # (B, num_frames_to_predict_for, 1, 3)
        """
        10.18 업데이트, 결국 concat해서 넣었기 때문에 
        이 axisangle이랑 translation이 s->t 인지 t->s 인지는 이후 
        pipeline에 따라 달라짐(s->t라고 장담못함)
        둘다 2개의 axisangle, translation 중 첫번째를 넣음
        
        target->source(t->t')
        """
        outputs[("axisangle", 0, f_i)] = axisangle
        outputs[("translation", 0, f_i)] = translation

        """
        invert는 
        [-1, 0]이 들어갔을 때 True
        [0, 1]이 들어갔을 때 False

        """
        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        # outputs[("cam_T_cam", 0, f_i)]이 target->source 의 transformation matrix
        # source가 t'(t' frame)이고 target이 t(t frame)
        # transformation matrix는 target->source  T_t->t'
        # 얘로 target(t)을 source(t')관점에서 본 이미지 좌표를 끌어낸 후
        # grid sampling 을 통해 source 관점에서 본 이미지를 target 관점으로 reprojection(I_t'->t)
        return outputs

    def val(self):
        self.set_eval()
        # 앞에서 정의해준건데?
        # Iteration이 멈추면?
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_image_pred(self, inputs, outputs):
        # Generate reprojected color image
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            # 192,640(입력 해상도) 로 변경
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False
            )
            source_scale = 0

            # Depth decoder가 예측하는건 disparity
            # -> single image 넣었는데 왜 depth 바로 안끌어내고 disp로 한다음에 변환?
            _, depth = disp_to_depth(
                disp, self.opt.min_depth, self.opt.max_depth
            )

            outputs[("depth", 0, scale)] = depth
            #-1, 1
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                # 여기서 0은 scale
                T = outputs[("cam_T_cam", 0, frame_id)]

                # 2D depth map -> 3D point
                # (B, 4, hw) : X, Y, Z, 1

                # I_{t}의 깊이 정보인 D_t (depth)를 3D 좌표로 backprojection

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )

                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )

                # World coordinate와 camera coordinate를 같게 보면 가능
                # 애초에 Cam point 라고 변수를 놓은거부터가 두개를 구분하지 않겠다는 뜻으로 생각
                # scale 된 해상도임
                outputs[("sample", frame_id, scale)] = pix_coords

                # inputs[("color", frame_id, source_scale)] : I_{t'}의 원본 해상도 이미지
                # outputs[("sample", frame_id, scale)] : uv 좌표 (t')

                # outputs[("color", frame_id, scale)] : I_{t' -> t}
                # => sample 좌표 (uv 좌표)를 이용하여 I_{t'}를 I_{t}에 맞게 sampling 하여 생성 (I_{t'->t})하고
                #    최종적으로 I_{t'->t}와 I_{t}의 reprojection loss가 낮아지도록 학습하므로
                #    학습 완료 시 I_{t' -> t}와 I_{t}는 유사해짐
                #    F.grid_sample 연산에 따라 기존 이미지 크기를 벗어난 sample 좌표들은 가장 외곽의 값으로 대체됨

                #(B, 3, H, W)
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    # (B, 3, H, W)
                    inputs[("color", frame_id, source_scale)],
                    # (B, H, W, 2)
                    outputs[("sample", frame_id, source_scale)],
                    padding_mode="border"
                )

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] =\
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        # pred.shape : (B, 3, source_height, source_width)
        # target.shape : (B, 3, source_height, source_width)
        # (B, 3, source_height, source_width)
        abs_diff = torch.abs(target-pred)
        # L1
        # (B, 1, source_height, source_width)
        l1_loss = abs_diff.mean(1, keepdims=True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85*ssim_loss+0.15*l1_loss
        # (B, 1, source_height, source_width)
        return reprojection_loss

    # Batch 단위로 넣어주는거
    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        # scale 별 loss 구하기
        for scale in self.opt.scales:
            loss = 0

            source_scale = 0

            # (B, C=1, H//(2**scale), W//(2**scale))
            disp = outputs[("disp", scale)]
            # (B, C=3, H//(2**scale), W//(2**scale)): Downscaled image
            color = inputs[("color", 0, scale)]
            # (B, C=3, 640, 192): Original size image
            target = inputs[("color", 0, source_scale)]

            reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                # outputs[("color", frame_id, scale)]의 크기는 source_scale을 가지나
                # scale 만큼 downsampling된 이미지로 부터 생성된 것을 의미함
                # Output은 reprojected image
                # (B, 3, H, W)
                pred = outputs[("color", frame_id, scale)]
            # pred.shape : (B, 3, source_height, source_width)
            # target.shape : (B, 3, source_height, source_width)
                reprojection_losses.append(
                    # pixel wise-> mean(dim=1) (B, 1, h, w)
                    self.compute_reprojection_loss(pred, target)
                )
            # (B, 2, h, w)
            reprojection_loss = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                # I_{t'} (pred) 와 I_{t} (target) 간의 reprojection loss를 구하며 이를 identity_reprojection_loss 라고 함
                # auto masking을 적용하기 위하여 사용함
                # 얘는 input끼리 loss 구하는거
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target)
                )
            identity_reprojection_loss = torch.cat(
                identity_reprojection_losses, 1)

            # (B, 2, h, w)
            # For tie breaking:
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device
            )*0.00001

            # (B, 4, h, w)
            combined = torch.cat(
                (identity_reprojection_loss, reprojection_loss), dim=1
            )
##
            # identity_reprojection_loss와 reprojection_loss 중 가장 작은 값을 선택합니다.
            # 만약 identity_reprojection_loss 중에서 가장 작은 부분이 선택된다면
            # 두 Frame 간 static 한 픽셀에 의해 loss가 가장 작아서 선택 된 것으로 가정하며
            # static한 픽셀은 차이가 거의 없어 Loss가 0에 가까워지므로 auto-masking이 됩니다.
            # reprojection_loss 중에서 가장 작은 값이 선택된다면 두 Frame 간 차이가 있는 것으로 가정하며,
            # occluded pixel 문제 또한 처리된 것으로 판단합니다.

            # (B, h, w)
            # (B, h, w)
            # 여기서는 4개를 비교함
            # (-1과 0의 reprojection error, 0과 1의 reprojection error): identity_reprojection_loss
            # (-1 -> 0, 0의 reprojection error,  1 -> 0과  0의 reprojection error): reprojection_loss
            # 0은 다들어감 input 안변함
            # idxs는 몇 번째 픽셀(Index)인지(0~3)
            to_optimize, idxs = torch.min(combined, dim=1)

            # (B, h, w)
            # minimum의 index가 identity shape보다 크면 reprojection loss 가 선택된거
            # Reprojection이 minimum일 때 1: 즉 잘 선택되고 마스크 없으면 1
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            # SSIM + l1
            loss += to_optimize.mean()

            # norm_disp : d^{*}_t 에 해당하며 norm_disp와 color 이미지를 통하여 smoothness loss를 구합니다.

            # disp: (B, C=1, H//(2**scale), W//(2**scale))
            # mean_disp: (B, 1, 1, 1)
            mean_disp = disp.mean(2, True).mean(3, True)
            # normalize
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            # Scale 별로 돌면서 더하기
            total_loss += loss
            # Scale 별 loss update
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80
        )
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        # Depth mask
        mask = depth_gt > 0

        crop_mask = torch.zeors_like(mask)
        # Size mask
        crop_mask[:, :, 153:371, 44:1179] = 1
        mask = mask*crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):

        samples_per_sec = self.opt.batch_size/duration
        # 이때까지 걸린 시간
        time_sofar = time.time()-self.start_time
        # 남은 시간
        # (전체 step/현재 step) * 현재 time
        training_time_left = (self.num_total_steps /
                              self.step-1.0)*time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(
            self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(
                self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
