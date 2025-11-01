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
    parser.add_argument('--dataset', type=str, default='PhoCAL', choices=['HAMMER', 'HouseCat6D', 'PhoCAL', 'TransCG', 'XYZ-IBD', 'YCB-V', 'T-LESS', 'GN-Trans'])
    parser.add_argument('--dataset_root', type=str, default='/data/robotarm/dataset')
    parser.add_argument('--split', type=str, default='/home/robotarm/object_depth_percetion/dataset/splits/PhoCAL_test.txt', help='Path to split file listing RGB images')
    parser.add_argument('--output_root', type=str, default='/data/robotarm/result/depth/mixed')
    parser.add_argument('--method', type=str, default='d3roma_zs_360x640')
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=5)
    parser.add_argument('--camera', type=str, default='d435', choices=['l515', 'd435', 'tof'])
    args = parser.parse_args()

    args.pred_only = True
    args.grayscale = True
    depth_factor = 1000.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    from utils.camera import Realsense
    camera = Realsense.default_real(args.camera)

    from inference import D3RoMa
    droma = D3RoMa([
        "task=eval_ldm_mixed_rgb+raw",
        "task.resume_pretrained=experiments/ldm_sf-241212.2.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.ddpm1000.Dreds_HssdIsaacStd_ClearPose.180x320.rgb+raw.w0.0/epoch_0056",
        "task.eval_num_batch=1",
        "task.image_size=[320,640]",
        "task.eval_batch_size=1",
        "task.num_inference_rounds=1",
        "task.num_inference_timesteps=10",
        "task.num_intermediate_images=5",
        "task.write_pcd=true"
    ], camera, variant="rgb+raw")

    # ===== 读取 split 文件中的 rgb 路径 =====
    with open(args.split, 'r') as f:
        rgb_lines = [line.strip().split()[0] for line in f if line.strip()]

    for rgb_rel_path in tqdm(rgb_lines):
        rgb_path = os.path.join(args.dataset_root, args.dataset, rgb_rel_path)
        depth_scale = 1.0
        # 推导出 raw depth 路径
        if args.dataset == 'HAMMER':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'polarization', f'depth_{args.camera}', f'{frame_id:06d}.png')
        elif args.dataset == 'HouseCat6D':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'PhoCAL':
            scene_name = rgb_rel_path.split('/')[0]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'TransCG':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-2])
            if args.camera == 'd435':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth1.png')
            elif args.camera == 'l515':
                depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id}', 'depth2.png')
        elif args.dataset == 'GN-Trans':
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(rgb_rel_path.split('/')[-1].split('_')[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'scenes', scene_name, f'{frame_id:04d}_depth_sim.png')
        elif args.dataset == 'XYZ-IBD':
            depth_scale = 0.09999999747378752
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'val', scene_name, 'depth_xyz', f'{frame_id:06d}.png')
        elif args.dataset == 'YCB-V':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test', scene_name, 'depth', f'{frame_id:06d}.png')
        elif args.dataset == 'T-LESS':
            depth_scale = 0.1
            scene_name = rgb_rel_path.split('/')[1]
            frame_id = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
            depth_path = os.path.join(args.dataset_root, args.dataset, 'test_primesense', scene_name, 'depth', f'{frame_id:06d}.png')
        if not os.path.exists(depth_path):
            print(f'[Warning] Raw depth not found: {depth_path}, skipping')
            continue

        try:
            raw = np.array(Image.open(depth_path)) * depth_scale * 1e-3
            rgb = np.array(Image.open(rgb_path))
            raw_height, raw_width, dim = rgb.shape[:3]
            # print(dim)
            if dim == 4:
                rgb = rgb[:, :, :3]
        except Exception as e:
            print(f"[Error] Failed to read {rgb_path} or {depth_path}: {e}")
            continue

        save_dir = os.path.join(args.output_root, args.dataset, args.method, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        pred_depth = droma.infer_with_rgb_raw(rgb, raw)
        metric_depth = (pred_depth * depth_factor).astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, f'{frame_id:06d}_depth.png'), metric_depth)
