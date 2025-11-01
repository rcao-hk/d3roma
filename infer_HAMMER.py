import os

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--dataset_root', type=str, default='/data/robotarm/dataset/HAMMER')
    parser.add_argument('--output_root', type=str, default='/data/robotarm/result/depth/mixed/hammer')
    parser.add_argument('--method', type=str, default='d3roma_zs_360x640')
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=2)
    parser.add_argument('--camera', type=str, default='d435', choices=['l515', 'd435', 'tof'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--with-step', dest='with_step', action='store_true', help='save depth with step')
    args = parser.parse_args()
    
    args.pred_only = True
    args.grayscale = True
    depth_factor = 1000.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    from utils.camera import Realsense
    camera = Realsense.default_real(args.camera)

    # from utils.camera import RGBDCamera
    # camera = RGBDCamera.default_hammer()
    overrides = [
        # uncomment if you choose variant left+right+raw
        # "task=eval_ldm_mixed",
        # "task.resume_pretrained=experiments/ldm_sf-mixed.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.nossi.my_ddpm1000.SceneFlow_Dreds_HssdIsaacStd.180x320.cond7-raw+left+right.w0.0/epoch_0199",
        
        # uncomment if you choose variant rgb+raw
        "task=eval_ldm_mixed_rgb+raw",
        "task.resume_pretrained=experiments/ldm_sf-241212.2.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.ddpm1000.Dreds_HssdIsaacStd_ClearPose.180x320.rgb+raw.w0.0/epoch_0056",

        # rest of the configurations
        "task.eval_num_batch=1",
        "task.image_size=[360,640]", 
        "task.eval_batch_size=1",
        "task.num_inference_rounds=1",
        "task.num_inference_timesteps=10", "task.num_intermediate_images=5",
        "task.write_pcd=true"
    ]
    """ if False: # turn on guidance
        overrides += [
            "task.sampler=my_ddpm", 
            "task.guide_source=raw-depth", 
            "task.flow_guidance_mode=gradient", 
            "task.flow_guidance_weights=[1.0]"
        ] """

    from inference import D3RoMa
    droma = D3RoMa(overrides, camera, variant="rgb+raw")

    # filenames = glob.glob(os.path.join(args.dataset_root, '*/*_color.png'))
    test_scenes = ['scene12_traj1_1', 'scene12_traj2_1','scene12_traj2_2', 'scene13_traj1_1', 'scene13_traj2_1', 'scene13_traj2_2', 'scene14_traj1_1', 'scene14_traj2_1', 'scene14_traj2_2']
    scenes = test_scenes

    for scene_name in scenes:
        # if 'scene13' not in scene_name:
            # continue
        # print(scene_name)
        # print("scene:{}".format(scene_name))
        data_root = os.path.join(args.dataset_root, scene_name)
        anno_len = len(os.listdir(os.path.join(args.dataset_root, scene_name, 'polarization', 'rgb')))
        
        for anno_idx in range(0, anno_len):
            # print(f'Progress {k+1}/{len(filenames)}: {filename}')
            # scene_name, _, image_name = filename.split('/')[-3:]
            # scene_name = filename.split('/')[0]
            # scene_root = os.path.relpath(filename, args.dataset_root)
            # scene_name = scene_root.split('/')[0]
            # image_name = scene_root.split('/')[-1]
            
            # image_name = os.path.splitext(image_name)[0]
            rgb_path = os.path.join(data_root, 'polarization', 'rgb/{:06d}.png'.format(anno_idx))
            gt_depth_path = os.path.join(data_root, 'polarization', '_gt/{:06d}.png'.format(anno_idx))
            depth_path = os.path.join(data_root, 'polarization', 'depth_{}/{:06d}.png'.format(args.camera, anno_idx))
        
            # raw_image = cv2.imread(rgb_path)
            # obs_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            raw = np.array(Image.open(depth_path)) * 1e-3
            rgb = np.array(Image.open(rgb_path))
    
            # image = cv2.resize(raw_image, (640, 360), interpolation=cv2.INTER_LINEAR)

            save_path = os.path.join(args.output_root, args.method, scene_name)
            os.makedirs(save_path, exist_ok=True)
            
            # image = raw_image
            # init_depth, depth = model.infer_image(image, obs_depth, args.input_size, True)

            # depth_aligned = camera.transform_depth_to_rgb_frame(raw) #if not alreay aligned
            # if True: # visualize aligned depth for realsense d415
            #     valid = (depth_aligned > args.min_depth) & (depth_aligned < args.max_depth)
            #     import matplotlib.pyplot as plt
            #     cmap_spectral = plt.get_cmap('Spectral')
            #     raw_depth_normalized = np.zeros_like(depth_aligned)
            #     raw_depth_normalized[valid] = (depth_aligned[valid] - depth_aligned[valid].min()) / (depth_aligned[valid].max() - depth_aligned[valid].min())
            #     Image.fromarray((cmap_spectral(raw_depth_normalized)*255.)[...,:3].astype(np.uint8)).save(f"raw_aligned.png")

            pred_depth = droma.infer_with_rgb_raw(rgb, raw)
        
            # rel_depth, depth = model.infer_image(image, args.input_size)
            
            metric_depth = pred_depth * depth_factor
            metric_depth = metric_depth.astype(np.uint16)

            cv2.imwrite(os.path.join(save_path, '{:06d}_depth.png'.format(anno_idx)), metric_depth)