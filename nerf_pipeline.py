#!/usr/bin/env python3
"""
End-to-end NeRF reconstruction pipeline (GPU/CPU-friendly + optional COLMAP)
==========================================================================
• 使用此脚本启动训练，避免直接调用 train.py 导致补丁失效。
• GPU 分支：检测到 CUDA 可用时，不做降级或补丁，使用全分辨率。
• CPU 分支：CUDA 不可用时，自动降级参数并补丁跳过过滤。
• `--skip_colmap`：跳过 COLMAP SfM，直接使用已有的 transforms*.json。
• 通过 `--train_arg KEY VAL` 透传其他 train.py 参数。
"""
from __future__ import annotations
import argparse, glob, json, math, os, pathlib, platform, shlex, shutil, subprocess, sys
import torch

# ---------- Defaults ----------
DEFAULT_IMAGES       = "./data/lego/images"
DEFAULT_TENSORF_REPO = "./TensoRF"
COLMAP_ENV           = os.environ.get("COLMAP_CMD", "colmap")

# ---------- Utilities ----------
def run(cmd: str, cwd: str | None = None, env: dict | None = None) -> None:
    print(f"\033[36m» {cmd}\033[0m")
    if subprocess.call(cmd, shell=True, cwd=cwd, env=env) != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def ensure_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def quote(path: str) -> str:
    return f'"{path}"' if (" " in path or os.name == "nt") else shlex.quote(path)

# ---------- Automatic patch for CPU-only filtering ----------
def patch_tensorbase(repo: str) -> None:
    tb = pathlib.Path(repo)/"models"/"tensorBase.py"
    if not tb.exists(): return
    text = tb.read_text(encoding="utf-8")
    if "return all_rays, all_rgbs" in text: return
    lines, out, in_class, patched = text.splitlines(), [], False, False
    for line in lines:
        out.append(line)
        stripped = line.lstrip()
        if stripped.startswith("class ") and "TensorBase" in stripped:
            in_class = True
        elif in_class and stripped.startswith("def filtering_rays") and not patched:
            indent = line[:line.find(stripped)] + "    "
            out.append(indent + "return all_rays, all_rgbs  # patched skip filtering")
            patched = True
        elif in_class and line and not line.startswith(" "):
            in_class = False
    tb.write_text("\n".join(out), encoding="utf-8")
    print(f"🩹 Patched TensorBase.filtering_rays in {tb}")

# ---------- COLMAP SfM ----------
def colmap_sfm(images: str, out_dir: str, colmap_cmd: str) -> str:
    db = os.path.join(out_dir, "database.db")
    sparse = ensure_dir(os.path.join(out_dir, "sparse"))
    exe = quote(colmap_cmd)
    steps = [
        f"{exe} feature_extractor --ImageReader.single_camera 1 --database_path {db} --image_path {images}",
        f"{exe} exhaustive_matcher --database_path {db}",
        f"{exe} mapper --database_path {db} --image_path {images} --output_path {sparse}",
        f"{exe} model_converter --input_path {sparse}/0 --output_path {sparse}/txt --output_type TXT"
    ]
    for s in steps: run(s)
    return os.path.join(sparse, "txt")

# ---------- COLMAP → NeRF JSON ----------
def colmap2json(txt_dir: str, images: str, repo: str, out_json: str) -> None:
    script = pathlib.Path(repo)/"tools"/"colmap2nerf.py"
    if not script.exists():
        raise FileNotFoundError("colmap2nerf.py missing in TensoRF/tools")
    run(f"python {script.as_posix()} --colmap_out {txt_dir} --images {images} --out {out_json}")

# ---------- Train wrapper ----------
def train_tensorf(datadir: str, repo: str, iters: int, batch: int, down: int, exp: str, extra: list[str]) -> str:
    # auto-install deps
    for pkg in ("configargparse","imageio","kornia","scipy","tensorboard"):
        try: __import__(pkg)
        except ImportError:
            print(f"⚙️ Installing missing dependency {pkg}...")
            run(f"{sys.executable} -m pip install --no-cache-dir {pkg}")
    # locate config
    cfg = pathlib.Path(repo)/"configs"/"lego.txt"
    if not cfg.exists(): cfg = pathlib.Path(repo)/"configs"/"nerf_synthetic"/"lego_cp.txt"
    if not cfg.exists(): raise FileNotFoundError(f"Config not found in {repo}/configs")
    cfg_abs = cfg.resolve().as_posix()
    cmd = [
        sys.executable, "train.py",
        "--config", cfg_abs,
        "--datadir", datadir,
        "--expname", exp,
        "--n_iters", str(iters),
        "--batch_size", str(batch),
        "--downsample_train", str(down)
    ] + extra
    run(" ".join(cmd), cwd=repo)
        # locate produced checkpoints by recursive search in datadir and repo/log
    ckpt_paths: list[pathlib.Path] = []
    # search datadir for .pth files
    for p in pathlib.Path(datadir).rglob("*.pth"):
        ckpt_paths.append(p)
    # search repo/log for .pth files
    log_root = pathlib.Path(repo)/"log"
    if log_root.exists():
        for p in log_root.rglob("*.pth"):
            ckpt_paths.append(p)
    # sort by modification time
    ckpt_paths = sorted(ckpt_paths, key=lambda p: p.stat().st_mtime)
    if not ckpt_paths:
        raise RuntimeError("train.py produced no checkpoint")
    # return the latest checkpoint
    if not ckpt_paths:
        raise RuntimeError("train.py produced no checkpoint")

    return str(ckpt_paths[-1])
    #return ckpts[-1]

#可视化
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 假设你pipeline里train_tensorf返回的是ckpt路径，训练统计数据保存路径叫 training_stats.json

def plot_training_results(stats_path: Path):
    if not stats_path.exists():
        print(f"⚠️ Stats file {stats_path} not found, skip plotting.")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    iterations = stats.get("iterations", list(range(len(stats.get("train_psnr", [])))))
    train_psnr = stats.get("train_psnr", [])
    test_psnr = stats.get("test_psnr", [])
    train_loss = stats.get("train_loss", [])

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(iterations, train_psnr, label="Train PSNR")
    if test_psnr:
        plt.plot(iterations, test_psnr, label="Test PSNR")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.legend()
    plt.title("PSNR Curve")

    plt.subplot(1,2,2)
    plt.plot(iterations, train_loss, label="Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")

    plt.tight_layout()
    out_file = stats_path.parent / "training_plot.png"
    plt.savefig(out_file)
    plt.close()
    print(f"📈 Training plots saved to {out_file}")

# ---------- Main pipeline ----------
def main() -> None:
    ap = argparse.ArgumentParser("NeRF pipeline")
    ap.add_argument("--obj_name", required=True)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--tensorf_repo", default=DEFAULT_TENSORF_REPO)
    ap.add_argument("--colmap_cmd", default=COLMAP_ENV)
    ap.add_argument("--skip_colmap", action="store_true")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--train_arg", nargs=2, action="append", default=[], metavar=("KEY","VAL"))
    args = ap.parse_args()
    if torch.cuda.is_available():
        print("⚙️ GPU mode detected, full settings", vars(args))
    else:
        print("⚙️ CPU mode, applying patch & parameter limits", vars(args))
        args.iters      = min(args.iters,      10000)
        args.batch_size = min(args.batch_size, 512)
        args.downsample = max(args.downsample, 16)
        patch_tensorbase(args.tensorf_repo)
    ws = ensure_dir(os.path.join("workspace", args.obj_name))
    if args.skip_colmap:
        parent = pathlib.Path(args.images).resolve().parent
        found=False
        for name in ("transforms_train.json","transforms_val.json","transforms_test.json","transforms.json"):
            src = parent/name
            if src.exists(): shutil.copy(src, pathlib.Path(ws)/name); found=True
        if not found: raise RuntimeError("transforms*.json not found next to images")
        datadir = parent.as_posix()
    else:
        txt = colmap_sfm(args.images, ws, args.colmap_cmd)
        colmap2json(txt, args.images, args.tensorf_repo, os.path.join(ws, "transforms.json"))
        datadir = ws
    extra: list[str] = []
    for k,v in args.train_arg:
        key = k if k.startswith("--") else f"--{k}"
        extra += [key, v]
    ckpt = train_tensorf(datadir, args.tensorf_repo, args.iters, args.batch_size, args.downsample, args.obj_name, extra)

    stats_path = Path("workspace") / args.obj_name / "training_stats.json"
    plot_training_results(stats_path)
    cams=[{"transform_matrix":[[math.cos(2*math.pi*i/120),0,-math.sin(2*math.pi*i/120),0],[0,1,0,0],[math.sin(2*math.pi*i/120),0,math.cos(2*math.pi*i/120),4],[0,0,0,1]]} for i in range(120)]
    json.dump({"camera_path":cams}, open(pathlib.Path(ws)/"turntable.json","w"), indent=2)
    # === 方案一：复制 transforms_*.json 到 checkpoint 目录 ===
    log_dir = ckpt_abs.parent
    for name in ("transforms_train.json", "transforms_val.json", "transforms_test.json"):
        src = pathlib.Path(ws) / name
        dst = log_dir / name
        if src.exists():
            shutil.copy(src, dst)
            print(f"✅ Copied {name} to {dst}")
        else:
            print(f"⚠️ Warning: {src} not found, skipping copy.")

    ckpt_abs = pathlib.Path(ckpt).resolve() # 拼成绝对路径
    render_path_abs = pathlib.Path(ws) / "turntable.json"

    run(f"{sys.executable} render.py -ckpt {ckpt_abs} -render_path {render_path_abs} -out _frames --fps 30", cwd=args.tensorf_repo)

    run(f"ffmpeg -y -r 30 -i _frames/%04d.png -vcodec libx264 -pix_fmt yuv420p {ws}/turntable.mp4")
    shutil.rmtree("_frames", ignore_errors=True)
    print("🎉 Done! Video saved at:", os.path.join(ws, "turntable.mp4"))

if __name__=="__main__":
    print("Python", platform.python_version(), "| CUDA", torch.cuda.is_available())
    main()


