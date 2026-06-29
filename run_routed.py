"""
TripoSR with Zero-Shot Classifier Routing
==========================================
Pipeline:
  Input Image → CLIP Zero-Shot Classifier → Pick Best Adapter → Hot-Swap LoRA → Run Inference → Save Mesh

Usage:
  python run_routed.py --image examples/Chair1.png
  python run_routed.py --image examples/Chair1.png --output_dir output/
  python run_routed.py --image examples/Chair1.png --no_remove_bg
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
from tsr.models.transformer.lora import LoRAAdapterManager


# ── Adapter registry ─────────────────────────────────────────────────────
# Maps each adapter file to the furniture categories it was trained on.
# The classifier picks from ALL category names, then we look up which
# adapter covers that category.

ADAPTER_CATEGORIES = {
    "lora_chair_bench.pt":          ["chair", "bench"],
    "lora_table_desk.pt":           ["table", "desk"],
    "lora_sofa_stool.pt":           ["sofa", "stool"],
    "lora_bed_wardrobe.pt":         ["bed", "wardrobe"],
    "lora_cabinet_bookshelf.pt":    ["cabinet", "bookshelf"],
}


def build_category_to_adapter(adapter_dir):
    """Build a mapping from category name → adapter file path.
    Only includes adapters that actually exist on disk."""
    category_map = {}   # "chair" → "lora_chair_bench.pt"
    available = {}      # adapter_name → full_path

    for adapter_file, categories in ADAPTER_CATEGORIES.items():
        full_path = os.path.join(adapter_dir, adapter_file)
        if os.path.isfile(full_path):
            available[adapter_file] = full_path
            for cat in categories:
                category_map[cat] = adapter_file
        else:
            print(f"  [skip] {adapter_file} not found in {adapter_dir}")

    return category_map, available


# ── Image preprocessing ──────────────────────────────────────────────────

def preprocess_image(path, remove_bg=True):
    """Load and preprocess image (same as run.py)."""
    raw = Image.open(path)
    if not remove_bg:
        return raw.convert("RGB")
    try:
        import rembg
        from tsr.utils import remove_background, resize_foreground
        session = rembg.new_session()
        img = remove_background(raw, session)
        img = resize_foreground(img, 0.85)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255).astype(np.uint8))
    except ImportError:
        print("  rembg not installed, using raw image")
        return raw.convert("RGB")


# ── CLIP zero-shot classification ────────────────────────────────────────

_clip_classifier = None

CATEGORY_TO_LABEL = {
    "chair": "a chair",
    "bench": "a bench",
    "table": "a table",
    "desk": "a desk",
    "sofa": "a sofa",
    "stool": "a stool",
    "bed": "a bed",
    "wardrobe": "a wardrobe",
    "cabinet": "a cabinet",
    "bookshelf": "a bookshelf",
}

def classify_image(image, candidate_labels, device):
    """Use CLIP zero-shot classification to determine furniture category.
    The model is loaded once and cached for subsequent calls."""
    global _clip_classifier
    from transformers import pipeline as hf_pipeline

    if _clip_classifier is None:
        print("  Loading CLIP zero-shot classifier...")
        _clip_classifier = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=0 if device == "cuda" else -1,
        )

    clip_labels = [CATEGORY_TO_LABEL.get(cat, f"a {cat}") for cat in candidate_labels]
    label_to_category = dict(zip(clip_labels, candidate_labels))
    results = _clip_classifier(image.convert("RGB"), candidate_labels=clip_labels)
    category_scores = {
        label_to_category[item["label"]]: float(item["score"])
        for item in results
        if item["label"] in label_to_category
    }

    # Find the top predicted category
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]

    print(f"\n  Classification scores:")
    for cat, score in category_scores.items():
        bar = "#" * int(score * 40)
        print(f"    {cat:<15} {score:.3f}  {bar}")

    return best_category, best_score



# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TripoSR with routed LoRA adapters")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--adapter_dir", default="adapters", help="Directory containing .pt adapter files")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--pretrained", default="stabilityai/TripoSR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mc_resolution", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--no_remove_bg", action="store_true", help="Skip background removal")
    parser.add_argument("--bake_texture", action="store_true", default=True,
                        help="Bake texture atlas and export as GLB with embedded texture (default: True)")
    parser.add_argument("--no_bake_texture", dest="bake_texture", action="store_false",
                        help="Skip texture baking, export vertex-color OBJ instead")
    parser.add_argument("--texture_resolution", type=int, default=2048,
                        help="Texture atlas resolution in pixels (default: 2048)")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use BF16/FP16 autocast during encode (default: True)")
    parser.add_argument("--no_mixed_precision", dest="mixed_precision", action="store_false",
                        help="Disable mixed precision, run in FP32")
    parser.add_argument("--adaptive_mc", action="store_true", default=True,
                        help="Use adaptive marching cubes for faster mesh extraction (default: True)")
    parser.add_argument("--no_adaptive_mc", dest="adaptive_mc", action="store_false",
                        help="Use standard full-grid marching cubes")
    parser.add_argument("--mc_coarse_resolution", type=int, default=64,
                        help="Coarse grid resolution for adaptive MC (default: 64)")
    parser.add_argument("--category", default=None,
                        help="Skip CLIP and force a category directly, e.g. --category chair")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Pick best available half-precision format
    if args.mixed_precision and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Mixed precision: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")
    else:
        amp_dtype = None

    # ── 1. Discover available adapters ────────────────────────────────────
    print("Scanning adapters...")
    category_map, available_adapters = build_category_to_adapter(args.adapter_dir)

    if not available_adapters:
        print(f"ERROR: No adapter files found in {args.adapter_dir}/")
        print(f"Expected files: {list(ADAPTER_CATEGORIES.keys())}")
        sys.exit(1)

    candidate_labels = list(category_map.keys())
    print(f"  Available adapters: {len(available_adapters)}")
    print(f"  Categories: {candidate_labels}")

    # ── 2. Load and preprocess image ──────────────────────────────────────
    print(f"\nLoading image: {args.image}")
    image = preprocess_image(args.image, remove_bg=not args.no_remove_bg)
    print(f"  Size: {image.size[0]}x{image.size[1]}")

    # ── 3. Classify the image (or use forced category) ────────────────────
    if args.category:
        if args.category not in category_map:
            print(f"ERROR: --category '{args.category}' not recognised.")
            print(f"  Valid categories: {candidate_labels}")
            sys.exit(1)
        category, confidence, t_classify = args.category, 1.0, 0.0
        print(f"\n  >> Category: {category} (forced via --category)")
    else:
        print("\nClassifying with CLIP zero-shot...")
        t0 = time.perf_counter()
        category, confidence = classify_image(image, candidate_labels, device)
        t_classify = time.perf_counter() - t0
        print(f"\n  >> Category: {category} (confidence: {confidence:.3f})")
        print(f"  >> Classification time: {t_classify:.3f}s")

    adapter_file = category_map[category]
    print(f"  >> Adapter:  {adapter_file}")

    # ── 4. Load TripoSR model ─────────────────────────────────────────────
    print("\nLoading TripoSR model...")
    model = TSR.from_pretrained(args.pretrained, "config.yaml", "model.ckpt")

    # Find transformer and inject LoRA
    transformer = None
    for name, module in model.named_modules():
        if "Transformer1D" in type(module).__name__:
            transformer = module
            break

    if transformer is None:
        print("ERROR: Could not find Transformer1D in model")
        sys.exit(1)

    for param in model.parameters():
        param.requires_grad = False
    transformer.enable_lora(r=args.lora_r, alpha=args.lora_alpha)
    model.to(device)
    model.renderer.set_chunk_size(8192)
    model.eval()

    # ── 5. Set up adapter manager and load the chosen adapter ─────────────
    manager = LoRAAdapterManager(model, {
        name: path for name, path in available_adapters.items()
    })

    t0 = time.perf_counter()
    manager.set_adapter(adapter_file)
    t_swap = time.perf_counter() - t0

    # ── 6. Run inference ──────────────────────────────────────────────────
    sys.stdout.write("\nRunning inference...\n")
    sys.stdout.flush()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad()
    with torch.no_grad():
        with ctx:
            scene_codes = model([image], device=device)
    # Cast back to FP32 — autocast may leave scene_codes in BF16/FP16, which
    # breaks bake_texture and query_triplane (grid_sample requires matching dtypes)
    scene_codes = scene_codes.float()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_encode = time.perf_counter() - t0

    # Mesh extraction
    t0 = time.perf_counter()
    if args.adaptive_mc:
        from adaptive_mc import extract_mesh_adaptive
        mesh = extract_mesh_adaptive(
            model,
            scene_codes[0],
            device=device,
            resolution=args.mc_resolution,
            coarse_resolution=args.mc_coarse_resolution,
            has_vertex_color=not args.bake_texture,
        )
    else:
        meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
        mesh = meshes[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_mesh = time.perf_counter() - t0

    # ── 7. Save outputs ───────────────────────────────────────────────────
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(args.output_dir, f"{image_name}_routed")
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, "input.png"))

    t_bake = 0.0
    if args.bake_texture:
        import xatlas
        import trimesh
        from tsr.bake_texture import bake_texture

        sys.stdout.write("Baking texture atlas...\n")
        sys.stdout.flush()
        t0 = time.perf_counter()
        bake_output = bake_texture(mesh, model, scene_codes[0], args.texture_resolution)
        t_bake = time.perf_counter() - t0

        texture_img = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)

        obj_path = os.path.join(save_dir, "_tmp_mesh.obj")
        tex_path = os.path.join(save_dir, "_tmp_texture.png")
        texture_img.save(tex_path)
        xatlas.export(
            obj_path,
            mesh.vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            mesh.vertex_normals[bake_output["vmapping"]],
        )

        material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
        vis = trimesh.visual.TextureVisuals(uv=bake_output["uvs"], material=material)
        textured = trimesh.Trimesh(
            vertices=mesh.vertices[bake_output["vmapping"]],
            faces=bake_output["indices"],
            vertex_normals=mesh.vertex_normals[bake_output["vmapping"]],
            visual=vis,
            process=False,
        )
        glb_path = os.path.join(save_dir, "mesh.glb")
        textured.export(glb_path)

        for tmp in [obj_path, tex_path, obj_path.replace(".obj", ".mtl")]:
            if os.path.isfile(tmp):
                os.remove(tmp)

        output_file = "mesh.glb"
        output_desc = "GLB with embedded texture"
    else:
        obj_path = os.path.join(save_dir, "mesh.obj")
        mesh.export(obj_path)
        output_file = "mesh.obj"
        output_desc = "OBJ with vertex colors"

    # ── 8. Print summary ─────────────────────────────────────────────────
    precision_str = ("BF16" if amp_dtype == torch.bfloat16 else "FP16") if amp_dtype else "FP32"
    mc_str = f"Adaptive (coarse={args.mc_coarse_resolution})" if args.adaptive_mc else "Standard"
    verts = len(np.array(mesh.vertices))
    faces = len(np.array(mesh.faces))

    lines = [
        "",
        "=" * 60,
        "  ROUTED INFERENCE RESULTS",
        "=" * 60,
        f"  Image:          {args.image}",
        f"  Category:       {category} ({confidence:.1%} confidence)",
        f"  Adapter:        {adapter_file}",
        f"  Precision:      {precision_str}",
        f"  MC mode:        {mc_str}",
        "",
        f"  Classification: {t_classify:.3f}s",
        f"  Adapter swap:   {t_swap*1000:.1f}ms",
        f"  Encode:         {t_encode:.3f}s",
        f"  Mesh extract:   {t_mesh:.3f}s",
    ]
    if args.bake_texture:
        lines.append(f"  Texture bake:   {t_bake:.3f}s")
    lines += [
        f"  Total pipeline: {t_classify + t_swap + t_encode + t_mesh + t_bake:.3f}s",
        "",
        f"  Vertices:       {verts:,}",
        f"  Faces:          {faces:,}",
        f"  Output:         {output_desc}",
        "",
        f"  Saved to: {save_dir}/",
        f"    input.png",
        f"    {output_file}",
        "=" * 60,
    ]
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
