"""
TripoSR Benchmark
=================
Two modes:
  Base only:   python benchmark.py --mode base --image examples/chair.png
  With LoRA:   python benchmark.py --mode lora --image examples/chair.png --lora_path adapters/lora_chair.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsr.system import TSR


# ── Helpers ───────────────────────────────────────────────────────────────

def preprocess_image(path):
    """Load image, optionally remove background."""
    img = Image.open(path).convert("RGB")
    try:
        from tsr.utils import remove_background, resize_foreground
        import rembg
        session = rembg.new_session()
        img = remove_background(img, session)
        img = resize_foreground(img, 0.85)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255).astype(np.uint8))
    except ImportError:
        pass
    return img


def timed(fn, *args, **kwargs):
    """Run fn with CUDA-synced timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - t0


def mesh_stats(vertices, faces):
    """Basic mesh quality metrics."""
    stats = {"vertices": len(vertices), "faces": len(faces)}
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        stats["surface_area"] = float(m.area)
        stats["watertight"] = bool(m.is_watertight)
        stats["bounding_box_vol"] = float(m.bounding_box.volume)
        if m.is_watertight:
            stats["volume"] = float(m.volume)
    except ImportError:
        pass
    return stats


def run_inference(model, image, device, mc_resolution=256):
    """Run one full inference pass, return metrics dict + mesh data."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with torch.no_grad():
        scene_codes, t_encode = timed(model, [image], device)
        extract_fn = lambda: model.extract_mesh(scene_codes, True, resolution=mc_resolution)
        meshes, t_mesh = timed(extract_fn)

    mesh = meshes[0]
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    return {
        "t_encode": t_encode,
        "t_mesh": t_mesh,
        "t_total": t_encode + t_mesh,
        "vram_mb": vram,
        "mesh": mesh_stats(verts, faces),
        "mesh_obj": mesh,
        "verts": verts,
        "faces": faces,
    }


def print_results(label, result):
    """Print metrics table for a single run."""
    lines = [
        f"\n{'=' * 60}",
        f"  {label}",
        f"{'=' * 60}",
    ]
    if torch.cuda.is_available():
        lines.append(f"  GPU: {torch.cuda.get_device_name(0)}")
    lines.append("")

    rows = [
        ("Encode time (s)",     f"{result['t_encode']:.3f}"),
        ("Mesh extraction (s)", f"{result['t_mesh']:.3f}"),
        ("Total time (s)",      f"{result['t_total']:.3f}"),
        ("Peak VRAM (MB)",      f"{result['vram_mb']:.0f}"),
        ("Vertices",            f"{result['mesh']['vertices']:,}"),
        ("Faces",               f"{result['mesh']['faces']:,}"),
    ]
    if "surface_area" in result["mesh"]:
        rows.append(("Surface area", f"{result['mesh']['surface_area']:.4f}"))
    if "watertight" in result["mesh"]:
        rows.append(("Watertight", str(result["mesh"]["watertight"])))
    if result["mesh"].get("volume") is not None:
        rows.append(("Volume", f"{result['mesh']['volume']:.4f}"))

    lines.append(f"  {'Metric':<30} {'Value':>15}")
    lines.append(f"  {'-' * 47}")
    for name, val in rows:
        lines.append(f"  {name:<30} {val:>15}")
    lines.append(f"{'=' * 60}")

    print("\n".join(lines), flush=True)


# ── Mode: base ───────────────────────────────────────────────────────────

def run_base(model, image, device, mc_resolution):
    """Run the base TripoSR model and print metrics."""
    print("\nRunning BASE model inference...")
    result = run_inference(model, image, device, mc_resolution)
    print_results("BASE MODEL RESULTS", result)
    return result


# ── Mode: lora ───────────────────────────────────────────────────────────

def run_lora(model, image, device, mc_resolution, lora_path, lora_r, lora_alpha):
    """Run with LoRA adapter and print metrics."""
    from tsr.models.transformer.lora import load_lora_state_dict

    # Find transformer backbone
    transformer = None
    for name, module in model.named_modules():
        if "Transformer1D" in type(module).__name__:
            transformer = module
            break

    if transformer is None:
        print("ERROR: Could not find Transformer1D in model")
        sys.exit(1)

    # Freeze the ENTIRE model first (same as training script),
    # then enable LoRA which unfreezes only lora_A/lora_B params.
    print(f"\nInjecting LoRA (r={lora_r}, alpha={lora_alpha})...")
    for param in model.parameters():
        param.requires_grad = False
    transformer.enable_lora(r=lora_r, alpha=lora_alpha)
    model.to(device)  # move new LoRA params to GPU

    adapter_sd = torch.load(lora_path, map_location=device)
    load_lora_state_dict(model, adapter_sd)

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Adapter: {lora_path} ({os.path.getsize(lora_path) / 1e6:.1f} MB)")
    print(f"  LoRA params: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")

    print("\nRunning LoRA model inference...")
    result = run_inference(model, image, device, mc_resolution)
    print_results("LoRA ADAPTER RESULTS", result)
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TripoSR Benchmark")
    parser.add_argument("--mode", required=True, choices=["base", "lora"],
                        help="'base' = vanilla TripoSR, 'lora' = with LoRA adapter")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--lora_path", default=None, help="LoRA adapter .pt file (required for --mode lora)")
    parser.add_argument("--pretrained", default="stabilityai/TripoSR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mc_resolution", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--output_dir", default="output", help="Directory to save mesh and input image")
    args = parser.parse_args()

    if args.mode == "lora" and not args.lora_path:
        print("ERROR: --lora_path is required when --mode lora")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(args.pretrained, "config.yaml", "model.ckpt")
    model.to(device)
    model.renderer.set_chunk_size(8192)
    model.eval()

    image = preprocess_image(args.image)
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")

    if args.mode == "base":
        result = run_base(model, image, device, args.mc_resolution)
    else:
        result = run_lora(model, image, device, args.mc_resolution,
                          args.lora_path, args.lora_r, args.lora_alpha)

    # Save outputs
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(args.output_dir, f"{image_name}_{args.mode}")
    os.makedirs(save_dir, exist_ok=True)

    # Save input image
    input_path = os.path.join(save_dir, "input.png")
    image.save(input_path)

    # Save mesh
    mesh_path = os.path.join(save_dir, "mesh.obj")
    result["mesh_obj"].export(mesh_path)

    print(f"\nSaved to: {save_dir}/", flush=True)
    print(f"  input.png  - preprocessed input image", flush=True)
    print(f"  mesh.obj   - extracted 3D mesh", flush=True)


if __name__ == "__main__":
    main()
