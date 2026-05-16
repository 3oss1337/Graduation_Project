"""
SAM Segmentation — prompted object extraction for TripoSR
==========================================================
Segments a single object from an image using SAM with either a point or box
prompt, crops it onto a white background, and saves it ready for run_routed.py.

Setup (one-time):
    pip install segment-anything
    # checkpoint is auto-downloaded on first run

Usage:
    python segment.py --image photo.jpg --point 400,300
    python segment.py --image photo.jpg --box 100,50,500,400
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
PADDING_RATIO = 0.10   # 10% padding around the cropped object


# ── Checkpoint download ───────────────────────────────────────────────────

def ensure_checkpoint():
    if os.path.isfile(CHECKPOINT_PATH):
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Downloading SAM checkpoint → {CHECKPOINT_PATH}")
    print("  (375 MB, one-time download)")

    def progress(count, block_size, total):
        pct = min(count * block_size / total * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH, reporthook=progress)
    print()  # newline after progress bar
    print("  Download complete.")


# ── SAM inference ─────────────────────────────────────────────────────────

def run_sam(image_rgb, point=None, box=None, device="cuda"):
    """Run SAM predictor with a point or box prompt.

    Args:
        image_rgb: np.ndarray (H, W, 3) uint8
        point: (x, y) tuple or None
        box: (x1, y1, x2, y2) tuple or None
        device: "cuda" or "cpu"

    Returns:
        best_mask: np.ndarray (H, W) bool
        best_score: float
    """
    from segment_anything import SamPredictor, sam_model_registry

    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)

    print("Setting image...")
    predictor.set_image(image_rgb)

    # Build prompt arrays
    point_coords = None
    point_labels = None
    box_arr = None

    if point is not None:
        point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)   # 1 = foreground

    if box is not None:
        box_arr = np.array(box, dtype=np.float32)      # (x1, y1, x2, y2)

    print("Running prediction...")
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=True,   # returns 3 masks; we pick highest score
    )

    # Pick highest-confidence mask
    best_idx = int(np.argmax(scores))
    return masks[best_idx], float(scores[best_idx]), scores


# ── Image processing ──────────────────────────────────────────────────────

def mask_to_bbox(mask):
    """Return (x1, y1, x2, y2) tight bounding box of a boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)


def crop_on_white(image_rgb, mask, padding_ratio=PADDING_RATIO):
    """Crop the masked object, pad it, and place it on a white background.

    Returns:
        PIL.Image in RGB with white background, square crop.
    """
    H, W = mask.shape
    x1, y1, x2, y2 = mask_to_bbox(mask)

    obj_w = x2 - x1
    obj_h = y2 - y1
    pad_x = int(obj_w * padding_ratio)
    pad_y = int(obj_h * padding_ratio)

    # Padded crop bounds (clamped to image)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)

    # Crop image and mask
    crop_img  = image_rgb[cy1:cy2, cx1:cx2]
    crop_mask = mask[cy1:cy2, cx1:cx2]

    # Composite: object pixels kept, background → white (255)
    result = np.ones_like(crop_img) * 255
    result[crop_mask] = crop_img[crop_mask]

    # Make square (pad shorter side with white) so TripoSR sees a centered object
    h, w = result.shape[:2]
    side = max(h, w)
    square = np.ones((side, side, 3), dtype=np.uint8) * 255
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = result

    return Image.fromarray(square)


def make_preview(image_rgb, mask):
    """Overlay the mask on the original image as a semi-transparent highlight."""
    overlay = image_rgb.copy().astype(np.float32)

    # Darken non-masked region
    overlay[~mask] = overlay[~mask] * 0.4

    # Tint masked region green
    green_tint = np.array([80, 220, 100], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.6 + green_tint * 0.4

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw bounding box
    x1, y1, x2, y2 = mask_to_bbox(mask)
    preview = Image.fromarray(overlay)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(preview)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
    return preview


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM prompted segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--point", default=None,
                        help="Foreground point prompt as x,y  e.g. --point 400,300")
    parser.add_argument("--box", default=None,
                        help="Bounding box prompt as x1,y1,x2,y2  e.g. --box 100,50,500,400")
    parser.add_argument("--output_dir", default="output/segmented",
                        help="Directory to save cropped.png and preview.png")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    # Validate: need exactly one prompt
    if args.point is None and args.box is None:
        print("ERROR: provide either --point x,y  or  --box x1,y1,x2,y2")
        sys.exit(1)
    if args.point is not None and args.box is not None:
        print("ERROR: use --point OR --box, not both")
        sys.exit(1)

    # Parse prompt
    point = None
    box   = None
    if args.point:
        try:
            x, y = args.point.split(",")
            point = (int(x), int(y))
        except Exception:
            print(f"ERROR: --point must be x,y  got: {args.point}")
            sys.exit(1)

    if args.box:
        try:
            x1, y1, x2, y2 = args.box.split(",")
            box = (int(x1), int(y1), int(x2), int(y2))
        except Exception:
            print(f"ERROR: --box must be x1,y1,x2,y2  got: {args.box}")
            sys.exit(1)

    # Load image
    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    print(f"Loading image: {args.image}")
    pil_img   = Image.open(args.image).convert("RGB")
    image_rgb = np.array(pil_img)
    H, W      = image_rgb.shape[:2]
    print(f"  Size: {W}x{H}")

    if point:
        print(f"  Prompt: point ({point[0]}, {point[1]})")
    else:
        print(f"  Prompt: box ({box[0]}, {box[1]}) → ({box[2]}, {box[3]})")

    # Download checkpoint if needed
    ensure_checkpoint()

    # Run SAM
    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("  CUDA not available, falling back to CPU (will be slower)")

    mask, score, all_scores = run_sam(image_rgb, point=point, box=box, device=device)

    # Stats
    x1, y1, x2, y2 = mask_to_bbox(mask)
    area_px = int(mask.sum())
    print(f"\n  All mask scores:  {[f'{s:.3f}' for s in all_scores]}")
    print(f"  Best mask score:  {score:.4f}")
    print(f"  Bounding box:     ({x1}, {y1}) → ({x2}, {y2})")
    print(f"  Object area:      {area_px:,} px  ({100*area_px/(H*W):.1f}% of image)")

    # Build outputs
    os.makedirs(args.output_dir, exist_ok=True)

    cropped_path = os.path.join(args.output_dir, "cropped.png")
    preview_path = os.path.join(args.output_dir, "preview.png")

    cropped = crop_on_white(image_rgb, mask)
    cropped.save(cropped_path)

    preview = make_preview(image_rgb, mask)
    preview.save(preview_path)

    print(f"\n  Saved:")
    print(f"    {cropped_path}  ({cropped.size[0]}x{cropped.size[1]}, white bg)")
    print(f"    {preview_path}  (mask overlay)")
    print(f"\n  Next step:")
    print(f"    python run_routed.py --image {cropped_path}")


if __name__ == "__main__":
    main()
