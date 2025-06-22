# computer-vision-final
3D-Gaussian-vs-NeRF
# 3D Gaussian Splatting vs NeRF: Object Reconstruction and View Synthesis

## Task Overview

This project compares three approaches for 3D reconstruction and novel view synthesis:
1. Original NeRF
2. Accelerated NeRF (TensoRF)
3. 3D Gaussian Splatting

## Directory Structure

- `gaussian_pipeline.py`: Pipeline for 3D Gaussian Splatting
- `nerf_pipeline.py`: Pipeline for NeRF and TensoRF
- `workspace/`: Stores outputs, model checkpoints, renders, stats
- `data/train/`: Multi-view images of the object

## Setup

Make sure you have `conda` and `CUDA` installed.

```bash
# Create env
conda create -n gaussian python=3.10
conda activate gaussian

# Install dependencies
pip install -r requirements.txt
```

# Training
## Gaussian Splatting
```bash
python gaussian_pipeline.py --images_dir ./data/train \
                            --gaussian_repo ./gaussian-splatting \
                            --exp_name lego_demo2 \
```

## TensoRF / NeRF
```bash
python nerf_pipeline.py --obj_name lego_demo1 \
                        --images ./data/train \
                        --tensorf_repo ./TensoRF
```

# Rendering Notes (TensoRF)
The render.py script included in this project is used to generate turntable view renderings from trained models.

## Note:
This rendering script was manually added to the TensoRF directory. It is not part of the original TensoRF repository.
