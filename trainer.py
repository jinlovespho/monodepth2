# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import wandb 
from tqdm import tqdm
from torchvision.utils import save_image 
import random

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
            
        self.seed_everything(seed=42)

        # load data
        self.get_data()
        
        # load model
        self.get_model()
        
        # load optimizer and scheduler
        self.get_optimizer()
        
        # set projection 
        self.set_projection()
        
        # set logging tool(wandb,tensorboard)
        self.set_logging_tool()
        
        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_dataset), len(self.val_dataset)))

        self.save_opts()
    
    
    def seed_everything(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"seed : {seed}")
   
   
    def get_data(self):
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.train_dataset = self.dataset(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.val_dataset = self.dataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        
        self.train_loader = DataLoader(self.train_dataset, self.opt.batch_size, False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, self.opt.batch_size, False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
      
           
    def get_model(self):
        self.models={}
        self.params_to_train=[]
        
        if self.opt.model_name == 'monodepth2_baseline':
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.params_to_train += list(self.models["encoder"].parameters())

            breakpoint()
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.params_to_train += list(self.models["depth"].parameters())

            if self.use_pose_net:   # true
                if self.opt.pose_model_type == "separate_resnet":   # true
                    self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames)
                    self.models["pose_encoder"].to(self.device)
                    self.params_to_train += list(self.models["pose_encoder"].parameters())

                    self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
                    self.models["pose"].to(self.device)
                    self.params_to_train += list(self.models["pose"].parameters())

        
        elif self.opt.model_name == 'sf_vit_encB_decB':
            from networks.pho_models.vit import ViT
            from networks.pho_models.sf_vit import SF_ViT

            # load vit
            v = ViT(image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                    patch_size = 16,
                    num_classes = 1000,
                    dim = 768,
                    depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                    heads = 12,
                    mlp_dim = 3072).to(self.device)
        
            # load imagnet pretrained vit weights
            msg1=v.load_state_dict(torch.load("./pretrained_weights/vit_base_384.pth"))
            print(msg1)
                
            v.resize_pos_embed(192,640)

            breakpoint()
            self.models['depth'] = SF_ViT(  encoder=v,
                                            max_depth = self.opt.max_depth,
                                            features=[96, 192, 384, 768],           
                                            hooks=[2, 5, 8, 11],                  
                                            vit_features=768,                       
                                            use_readout='project').to(self.device)

            # load monodepth2 pose network
            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=2 ).to(self.device)
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(self.device)
            
            # set trainable params
            depth_params = [param for name, param in self.models['depth'].named_parameters()]
            depth_backbone_params = [param for name, param in self.models['depth'].named_parameters() if 'encoder' in name]
            else_params = [param for name, param in self.models['depth'].named_parameters() if 'encoder' not in name]
            else_params+=list(self.models['pose_encoder'].parameters())
            else_params+=list(self.models['pose'].parameters())
            
            self.params_to_train.append( {'params':depth_backbone_params, 'lr':self.opt.learning_rate/10} )
            self.params_to_train.append( {'params':else_params, 'lr':self.opt.learning_rate})
            
        elif self.opt.model_name == 'sf_croco_encB_decB':
            from networks.pho_models.vit_multiframe import ViT_Multiframe
            from networks.pho_models.sf_croco import SF_Croco

            # load vit
            v = ViT_Multiframe(image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                patch_size = 16,
                                num_classes = 1000,
                                dim = 768,
                                depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                heads = 12,
                                mlp_dim = 3072,).to(self.device)
            

            croco_weight = torch.load('./pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth')

            loaded_weight = {}
            
            for key, value in v.state_dict().items():
                if 'transformer' in key:
                    if '0.norm' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.norm1.{key.split(".")[-1]}']
                    elif 'qkv' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.attn.qkv.{key.split(".")[-1]}']
                    elif 'to_out' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.attn.proj.{key.split(".")[-1]}']
                    elif '1.norm' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.norm2.{key.split(".")[-1]}']
                    elif 'fn.net.0' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.mlp.fc1.{key.split(".")[-1]}']
                    elif 'fn.net.3' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.mlp.fc2.{key.split(".")[-1]}']
                    
                elif 'to_patch_embedding' in key:
                    loaded_weight[key] = croco_weight['model'][f'patch_embed.proj.{key.split(".")[-1]}']

                else:
                    print(key)
                    loaded_weight[key] = v.state_dict()[key]

            msg = v.load_state_dict(loaded_weight)
            print(msg)
            v.resize_pos_embed(192,640)
            
            breakpoint()
            self.models['depth'] = SF_Croco(    encoder=v,
                                                max_depth = self.opt.max_depth,
                                                features=[96, 192, 384, 768],           
                                                hooks=[2, 5, 8, 11],                  
                                                vit_features=768,                       
                                                use_readout='project').to(self.device)

            # load monodepth2 pose network
            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=2 ).to(self.device)
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(self.device)
            
            # set trainable params
            enc_params, enc_names=[],[]
            else_params, else_names=[],[]
            for name, param in self.models['depth'].named_parameters():
                if 'encoder' in name:
                    enc_params.append(param)
                    enc_names.append(name)
                else:
                    else_params.append(param)
                    else_names.append(name)
                    

            depth_params=list(self.models['depth'].parameters())
            assert len(depth_params) == len(enc_params) + len(else_params), 'CHECK TRAINABLE PARAMS !!'
            
            else_params+=list(self.models['pose_encoder'].parameters())
            else_params+=list(self.models['pose'].parameters())
            
            self.params_to_train.append( {'params':enc_params, 'lr':self.opt.learning_rate*0.1} )
            self.params_to_train.append( {'params':else_params, 'lr':self.opt.learning_rate})
            
            
          
        elif self.opt.model_name == 'mf_croco_encB_decB':
            from networks.croco_models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
            from networks.croco_models.pos_embed import interpolate_pos_embed
            from networks.croco_models.head_downstream import PixelwiseTaskWithDPT

            # Prepare model
            ckpt = torch.load(self.opt.pretrained_weight, 'cpu')
            croco_args = croco_args_from_ckpt(ckpt)
            croco_args['img_size'] = (self.opt.height, self.opt.width)
            print('Croco args: '+str(croco_args))
            self.opt.croco_args = croco_args # saved for test time 
            # prepare head 
            num_channels = 1
            # if self.opt.with_conf: num_channels += 1
            print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
            head = PixelwiseTaskWithDPT()
            head.num_channels = num_channels
            # build model and load pretrained weights
            breakpoint()
            self.models['depth'] = CroCoDownstreamBinocular(head, **croco_args).to(self.device)
            interpolate_pos_embed(self.models['depth'], ckpt['model'])
            msg = self.models['depth'].load_state_dict(ckpt['model'], strict=False)
            print(msg)
            
            # load monodepth2 pose network
            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=2 ).to(self.device)
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(self.device)
            
            # set trainable params
            enc_params, enc_names = [], []
            dec_params, dec_names = [], []
            else_params, else_names = [], []
            for name, param in self.models['depth'].named_parameters():
                if 'enc_blocks' in name:
                    enc_params.append(param)
                    enc_names.append(name)
                elif 'dec_blocks' in name:
                    dec_params.append(param)
                    dec_names.append(name)
                else:
                    else_params.append(param)
                    else_names.append(name)
            
            depth_params=list(self.models['depth'].parameters())
            assert len(depth_params) == len(enc_params) + len(dec_params) + len(else_params), 'CHECK TRAINABLE PARAMS !!'
            
            else_params+=list(self.models['pose_encoder'].parameters())
            else_params+=list(self.models['pose'].parameters())
            
            self.params_to_train.append( {'params':enc_params, 'lr':self.opt.learning_rate*0.1} )
            self.params_to_train.append( {'params':dec_params, 'lr':self.opt.learning_rate*0.1} )
            self.params_to_train.append( {'params':else_params, 'lr':self.opt.learning_rate})
        
        
        elif self.opt.model_name == 'mf_camap_croco_encB_decB':
            from networks.croco_models.camap_croco_downstream import CAMap_CroCoDownstreamBinocular, croco_args_from_ckpt
            from networks.croco_models.pos_embed import interpolate_pos_embed
            from networks.croco_models.camap_head_downstream import CAMap_PixelwiseTaskWithDPT
            
            # Prepare model
            ckpt = torch.load(self.opt.pretrained_weight, 'cpu')
            croco_args = croco_args_from_ckpt(ckpt)
            croco_args['img_size'] = (self.opt.height, self.opt.width)
            print('Croco args: '+str(croco_args))
            self.opt.croco_args = croco_args # saved for test time 
            # prepare head 
            num_channels = 1
            # if self.opt.with_conf: num_channels += 1
            print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
            head = CAMap_PixelwiseTaskWithDPT()
            head.num_channels = num_channels
            # build model and load pretrained weights
            breakpoint()
            self.models['depth'] = CAMap_CroCoDownstreamBinocular(head, **croco_args).to(self.device)
            interpolate_pos_embed(self.models['depth'], ckpt['model'])
            msg = self.models['depth'].load_state_dict(ckpt['model'], strict=False)
            print(msg)
            
            # load monodepth2 pose network
            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=2 ).to(self.device)
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(self.device)
            
            # set trainable params
            enc_params, enc_names = [], []
            dec_params, dec_names = [], []
            else_params, else_names = [], []
            for name, param in self.models['depth'].named_parameters():
                if 'enc_blocks' in name:
                    enc_params.append(param)
                    enc_names.append(name)
                elif 'dec_blocks' in name:
                    dec_params.append(param)
                    dec_names.append(name)
                else:
                    else_params.append(param)
                    else_names.append(name)
            
            depth_params=list(self.models['depth'].parameters())
            assert len(depth_params) == len(enc_params) + len(dec_params) + len(else_params), 'CHECK TRAINABLE PARAMS !!'
            
            else_params+=list(self.models['pose_encoder'].parameters())
            else_params+=list(self.models['pose'].parameters())
            
            self.params_to_train.append( {'params':enc_params, 'lr':self.opt.learning_rate*0.1} )
            self.params_to_train.append( {'params':dec_params, 'lr':self.opt.learning_rate*0.1} )
            self.params_to_train.append( {'params':else_params, 'lr':self.opt.learning_rate})
            
  
        else:
            pass
        
        if self.opt.load_weights_folder is not None:
            self.load_model()
            
        print("Training MODEL NAME:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
    
    
    def get_optimizer(self):
        self.model_optimizer = optim.Adam(self.params_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

    def set_projection(self):
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            
    def set_logging_tool(self):
        if self.opt.wandb:
            wandb.init(project = self.opt.wandb_proj_name,
                       name = self.opt.wandb_exp_name,
                       config = self.opt,
                       dir=self.log_path,
                       sync_tensorboard=True)       
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
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
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
          
        print("Training")
        self.set_train()
        tqdm_train = tqdm(self.train_loader, desc=f'Train Epoch: {self.epoch+1}/{self.opt.num_epochs}')
        for batch_idx, inputs in enumerate(tqdm_train):

            before_op_time = time.time()
            
            outputs, losses = self.process_batch(inputs)
            
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
                   
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        if self.opt.model_name == 'monodepth2_baseline':       
            features = self.models["encoder"](inputs["color", 0, 0])    # depth encoder
            outputs = self.models["depth"](features)                        # depth decoder
            
        elif self.opt.model_name.startswith('sf'):
            outputs=self.models['depth'](inputs)        
            
        elif self.opt.model_name.startswith('mf'):
            img1=inputs['color',0,0]
            img2=inputs['color',-1,0]
            do_mask=False
            outputs=self.models['depth'](img1,img2)
            # outputs=self.models['depth'](img1,img2, do_mask)
            
        outputs.update(self.predict_poses(inputs))  # update outputs dictionary with pose information        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:   # true
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:  # [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)   # pose decoder
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        # else:
        #     # Here we input all frames to the pose net (and predict all poses) together
        #     if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
        #         pose_inputs = torch.cat(
        #             [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

        #         if self.opt.pose_model_type == "separate_resnet":
        #             pose_inputs = [self.models["pose_encoder"](pose_inputs)]

        #     axisangle, translation = self.models["pose"](pose_inputs)

        #     for i, f_i in enumerate(self.opt.frame_ids[1:]):
        #         if f_i != "s":
        #             outputs[("axisangle", 0, f_i)] = axisangle
        #             outputs[("translation", 0, f_i)] = translation
        #             outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
        #                 axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)], padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
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
            depth_pred, [375, 1242], mode="bilinear", align_corners=True), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
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

                writer.add_image("disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:    # false
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:  # true
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
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
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
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
