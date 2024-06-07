import os 
import torch 
from torchvision.utils import save_image 
import numpy as np
import wandb 
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

def vis_input_imgs(inputs):
    
    vis_path='../vis/inputs'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    
    # color
    prev_img0 = inputs['color',-1,0]    # (b,3,192,640)
    prev_img1 = inputs['color',-1,1]    # (b,3,96,320)
    prev_img2 = inputs['color',-1,2]    # (b,3,48,160)
    prev_img3 = inputs['color',-1,3]    # (b,3,24,80)
    
    curr_img0 = inputs['color',0,0]
    curr_img1 = inputs['color',0,1]
    curr_img2 = inputs['color',0,2]
    curr_img3 = inputs['color',0,3]
    
    fut_img0 = inputs['color',1,0]
    fut_img1 = inputs['color',1,1]
    fut_img2 = inputs['color',1,2]
    fut_img3 = inputs['color',1,3]
    
    img0 = torch.cat([prev_img0, curr_img0, fut_img0], dim=2)
    img1 = torch.cat([prev_img1, curr_img1, fut_img1], dim=2)
    img2 = torch.cat([prev_img2, curr_img2, fut_img2], dim=2)
    img3 = torch.cat([prev_img3, curr_img3, fut_img3], dim=2)
    
    save_image(img0, f'{vis_path}/img0.jpg')
    save_image(img1, f'{vis_path}/img1.jpg')
    save_image(img2, f'{vis_path}/img2.jpg')
    save_image(img3, f'{vis_path}/img3.jpg')
    
    
    # color_aug
    prev_img0_aug = inputs['color_aug',-1,0]    # (b,3,192,640)
    prev_img1_aug = inputs['color_aug',-1,1]    # (b,3,96,320)
    prev_img2_aug = inputs['color_aug',-1,2]    # (b,3,48,160)
    prev_img3_aug = inputs['color_aug',-1,3]    # (b,3,24,80)
    
    curr_img0_aug = inputs['color_aug',0,0]
    curr_img1_aug = inputs['color_aug',0,1]
    curr_img2_aug = inputs['color_aug',0,2]
    curr_img3_aug = inputs['color_aug',0,3]
    
    fut_img0_aug = inputs['color_aug',1,0]
    fut_img1_aug = inputs['color_aug',1,1]
    fut_img2_aug = inputs['color_aug',1,2]
    fut_img3_aug = inputs['color_aug',1,3]
    
    img0_aug = torch.cat([prev_img0_aug, curr_img0_aug, fut_img0_aug], dim=2)
    img1_aug = torch.cat([prev_img1_aug, curr_img1_aug, fut_img1_aug], dim=2)
    img2_aug = torch.cat([prev_img2_aug, curr_img2_aug, fut_img2_aug], dim=2)
    img3_aug = torch.cat([prev_img3_aug, curr_img3_aug, fut_img3_aug], dim=2)
    
    save_image(img0_aug, f'{vis_path}/img0_aug.jpg')
    save_image(img1_aug, f'{vis_path}/img1_aug.jpg')
    save_image(img2_aug, f'{vis_path}/img2_aug.jpg')
    save_image(img3_aug, f'{vis_path}/img3_aug.jpg')

    # gt_depth
    save_image(inputs['depth_gt'], f'{vis_path}/depth_gt.jpg', normalize=True)
    
    print('VIS COMPLETE')

def vis_pred_depth_maps(inputs, outputs):
    original_height = 375
    original_width = 1242
    
    # input img resize
    color  = torch.nn.functional.interpolate(inputs['color',0,0], (original_height, original_width), mode="bilinear", align_corners=False)

    # disp resize
    disp = outputs[("disp", 0)] # (b,1,192,640)
    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
    

    num_pred_depth_sample=4
    for i in range(num_pred_depth_sample):
        wandb_val_dict={}
        val_result_imgs=[]
        
        # input_img
        input_img = color[i]
        input_img *= 255 
        
        # pred_depth
        disp_resized_np = disp_resized[i].squeeze().cpu().numpy()  # (b,1,192,640)->(192,640)
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_pred_depth = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        pred_depth = pil.fromarray(colormapped_pred_depth)
        
        # gt_depth
        gt_depth = inputs['depth_gt'][i].squeeze().cpu().numpy()
        vmax = np.percentile(gt_depth, 98)
        normalizer = mpl.colors.Normalize(vmin=gt_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_gt_depth = (mapper.to_rgba(gt_depth)[:, :, :3] * 255).astype(np.uint8)
        gt_depth = pil.fromarray(colormapped_gt_depth)
        
        val_result_imgs.append(wandb.Image(input_img, caption="Input Image"))
        val_result_imgs.append(wandb.Image(pred_depth, caption="Pred Depth"))
        val_result_imgs.append(wandb.Image(gt_depth, caption="GT Depth"))

        wandb_val_dict["validation depthmap"] = val_result_imgs
        wandb.log(wandb_val_dict)
        
        # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
        # im.save(name_dest_im)