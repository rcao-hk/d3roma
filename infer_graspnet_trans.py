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
    parser.add_argument('--dataset-root', type=str, default='/media/2TB/dataset/graspnet_sim/graspnet_trans')
    parser.add_argument('--output-root', type=str, default='/media/2TB/result/depth/graspnet_trans')
    parser.add_argument('--variant', type=str, default='rgb+raw')
    parser.add_argument('--split', type=str, default='test_similar', choices=['test', 'test_seen', 'test_novel', 'test_similar'])
    parser.add_argument('--input-width', type=int, default=320)
    parser.add_argument('--input-height', type=int, default=180)
    # parser.add_argument('--dataset', type=str, default='hypersim')
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=2)
    parser.add_argument('--eval-width', default=320, type=int)
    parser.add_argument('--eval-height', default=180, type=int)
    parser.add_argument('--camera', type=str, default='graspnet_d435')
    parser.add_argument('--mask', type=str, default='fore_mask', choices=['fore_mask', 'obj_mask'])
    parser.add_argument('--ckpt-epoch', type=int, default=None, help='checkpoint epoch to load')
    parser.add_argument('--latest-ckpt', action='store_true', help='use latest checkpoint')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--with-step', dest='with_step', action='store_true', help='save depth with step')
    parser.add_argument('--save-result', action='store_true', help='save results to output directory')
    args = parser.parse_args()
    
    args.pred_only = True
    args.grayscale = True
    depth_factor = 1000.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    from utils.camera import Realsense
    camera = Realsense.default_real(args.camera)
    if args.variant == 'rgb+raw':
        overrides = [            
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
    elif args.variant == 'left+right+raw':
        overrides = [            
            # uncomment if you choose variant rgb+raw
            "task=eval_ldm_mixed",
            "task.resume_pretrained=experiments/ldm_sf-mixed.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.nossi.my_ddpm1000.SceneFlow_Dreds_HssdIsaacStd.180x320.cond7-raw+left+right.w0.0/epoch_0199",

            # rest of the configurations
            "task.eval_num_batch=1",
            "task.image_size=[360,640]", 
            "task.eval_batch_size=1",
            "task.num_inference_rounds=1",
            "task.num_inference_timesteps=10", "task.num_intermediate_images=5",
            "task.write_pcd=true"
        ]
        
    from inference import D3RoMa
    droma = D3RoMa(overrides, camera, variant=args.variant)

    # image_list = glob.glob(os.path.join(args.dataset_root, '**', '*_color.png'))
    if args.split == 'test':
        scene_list = range(100, 190)
    elif args.split == 'test_seen':
        scene_list = range(100, 130)
    elif args.split == 'test_similar':
        scene_list = range(130, 160)
    elif args.split == 'test_novel':
        scene_list = range(160, 190)
    
    anno_sample_ratio = 0.1  # sample every anno_sample_ratio annotations
    
    for scene_idx in scene_list:
        for anno_idx in range(0, 256, int(1/anno_sample_ratio)):
            rgb_path = os.path.join(args.dataset_root, '{:05d}'.format(scene_idx), '{:04d}_color.png'.format(anno_idx))
    # for rgb_path in tqdm(image_list):
            # scene_idx = int(rgb_path.split('/')[-2])
            # anno_idx = int(rgb_path.split('/')[-1].split('_')[0])
            depth_path = rgb_path.replace('_color.png', '_depth_sim.png')
            left_path = rgb_path.replace('_color.png', '_ir_l.png')
            right_path = rgb_path.replace('_color.png', '_ir_r.png')
            
            left = np.array(Image.open(left_path))
            right = np.array(Image.open(right_path))
            raw = np.array(Image.open(depth_path)) * 1e-3
            rgb = np.array(Image.open(rgb_path))

            save_root = os.path.join(args.output_root, 'd3roma_{}'.format(args.variant), '{:05d}'.format(scene_idx))
            os.makedirs(save_root, exist_ok=True)
            
            # depth_aligned = camera.transform_depth_to_rgb_frame(raw) #if not alreay aligned
            # if True: # visualize aligned depth for realsense d415
            #     valid = (depth_aligned > args.min_depth) & (depth_aligned < args.max_depth)
            #     import matplotlib.pyplot as plt
            #     cmap_spectral = plt.get_cmap('Spectral')
            #     raw_depth_normalized = np.zeros_like(depth_aligned)
            #     raw_depth_normalized[valid] = (depth_aligned[valid] - depth_aligned[valid].min()) / (depth_aligned[valid].max() - depth_aligned[valid].min())
            #     Image.fromarray((cmap_spectral(raw_depth_normalized)*255.)[...,:3].astype(np.uint8)).save(f"raw_aligned.png")
            #     cv2.imwrite('rgb.png', rgb)

            if args.variant == 'rgb+raw':
                pred_depth = droma.infer_with_rgb_raw(rgb, raw)
            elif args.variant == "left+right+raw":
                pred_depth = droma.infer(left, right, raw, rgb)
                    
            metric_depth = pred_depth * depth_factor
            metric_depth = metric_depth.astype(np.uint16)

            cv2.imwrite(os.path.join(save_root, '{:06d}_depth.png'.format(anno_idx)), metric_depth)