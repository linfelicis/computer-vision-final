import argparse, json, os, subprocess, shutil
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_training_stats(stats_file: Path):
    if not stats_file.exists():
        print(f"⚠️ Stats file {stats_file} not found, skipping plotting.")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    iterations = stats.get("iterations", list(range(len(stats.get("train_loss", [])))))
    train_loss = stats.get("train_loss", [])
    train_psnr = stats.get("train_psnr", [])
    test_psnr = stats.get("test_psnr", [])

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(iterations, train_loss, label="Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(iterations, train_psnr, label="Train PSNR")
    if test_psnr:
        plt.plot(iterations, test_psnr, label="Test PSNR")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.title("PSNR over Iterations")
    plt.legend()

    plt.tight_layout()
    out_path = stats_file.parent / "training_plots.png"
    plt.savefig(out_path)
    plt.close()
    print(f"📈 Training plots saved to {out_path}")


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"\033[36m» {cmd}\033[0m")
    if subprocess.call(cmd, shell=True, cwd=cwd) != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def ensure_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def main():
    parser = argparse.ArgumentParser("3D Gaussian Pipeline")
    parser.add_argument("--images_dir", required=True, help="Path to input images")
    parser.add_argument("--gaussian_repo", required=True, help="Path to 3D Gaussian repo")
    parser.add_argument("--workspace", default="workspace", help="Output directory")
    parser.add_argument("--colmap_bin", default="colmap", help="COLMAP binary")
    parser.add_argument("--exp_name", default="gaussian_exp")
    parser.add_argument("--train_iters", type=int, default=3000)
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP if you already have transforms")
    args = parser.parse_args()

    ws = ensure_dir(f"{args.workspace}/{args.exp_name}")
    images = os.path.abspath(args.images_dir)

    if not args.skip_colmap:
        colmap_out = Path(ws)/"colmap"
        colmap_out.mkdir(parents=True, exist_ok=True)

        run(f"{args.colmap_bin} feature_extractor --ImageReader.single_camera 1 --database_path {colmap_out}/db.db --image_path {images}")
        run(f"{args.colmap_bin} exhaustive_matcher --database_path {colmap_out}/db.db")
        run(f"{args.colmap_bin} mapper --database_path {colmap_out}/db.db --image_path {images} --output_path {colmap_out}/sparse")
        run(f"{args.colmap_bin} model_converter --input_path {colmap_out}/sparse/0 --output_path {colmap_out}/sparse/txt --output_type TXT")

        colmap2nerf = Path(args.gaussian_repo)/"scripts"/"colmap2nerf.py"
        if not colmap2nerf.exists():
            raise FileNotFoundError("colmap2nerf.py not found in Gaussian repo/scripts")

        run(f"python {colmap2nerf} --colmap_out {colmap_out}/sparse/txt --images {images} --out {ws}/transforms.json")

    # 🛠️ 自动 fallback：尝试找现成的 transforms_train / test 替代 transforms.json
    transforms_json = Path(ws) / "transforms.json"
    if not transforms_json.exists():
        for alt_name in ["transforms_train.json", "transforms_test.json"]:
            alt_path = Path(ws) / alt_name
            if alt_path.exists():
                print(f"⚠️ transforms.json not found, using {alt_name} instead.")
                shutil.copy(alt_path, transforms_json)
                break

    if not transforms_json.exists():
        raise FileNotFoundError(f"transforms.json not found in {transforms_json}. Use --skip_colmap only when it exists or make sure fallback files exist.")

    train_script = Path("train.py")
    if not train_script.exists():
        raise FileNotFoundError("train.py not found in Gaussian repo")

    cmd = f"python {train_script.name} -s {images} -m {ws}/output --iterations {args.train_iters}"
    run(f"python {train_script} -s {images} -m {ws}/output --iterations {args.train_iters}", cwd=args.gaussian_repo)

    stats_path = Path(ws) / "training_stats.json"
    plot_training_stats(stats_path)


    render_script = Path(args.gaussian_repo)/"render.py"
    if not render_script.exists():
        raise FileNotFoundError("render.py not found in Gaussian repo")

    render_out = Path(ws)/"turntable.mp4"
    run(f"python {render_script} -m {ws}/output -o {render_out} -w 800 -h 800", cwd=args.gaussian_repo)

    print("🎉 Done! Video saved at:", render_out)

if __name__ == "__main__":
    main()
