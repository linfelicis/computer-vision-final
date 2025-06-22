# 文件名：TensoRF/render.py

import argparse
import torch
import os
import json
from pathlib import Path
from models.tensoRF import TensorVMSplit
from renderer import OctreeRender_trilinear_fast, evaluation_path
from dataLoader import blender

def load_render_path(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', type=str, required=True, help="Path to ckpt .pth file")
    parser.add_argument('-render_path', type=str, required=True, help="Path to turntable.json")
    parser.add_argument('-out', type=str, default='_frames', help="Output directory")
    parser.add_argument('--fps', type=int, default=30, help="FPS for rendered video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)

    # === 加载模型
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs['device'] = device
    tensorf = TensorVMSplit(**kwargs).to(device)
    tensorf.load(ckpt)

    # === 加载turntable路径
    print(f"Loading render path from {args.render_path}")
    render_poses = load_render_path(args.render_path)

    # === 设置假的 test_dataset，仅用来渲染（你用 Blender 数据集所以这里也用 blender）
    dataset = blender.BlenderDataset(
        datadir=str(ckpt_path.parent),  # 用 log 目录推断出数据目录
        split='test',  # 其实用不上图像本身
        downsample=1.0,
        is_stack=True
    )

    # === 开始渲染
    print(f"Rendering to {args.out}")
    evaluation_path(
        test_dataset=dataset,
        tensorf=tensorf,
        c2ws=render_poses,
        renderer=OctreeRender_trilinear_fast,
        savePath=args.out,
        white_bg=True,
        device=device
    )

if __name__ == '__main__':
    main()
