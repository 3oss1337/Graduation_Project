"""
FurnishAR Backend v3 — Hybrid pipeline
=======================================
Supports two frontend pipeline modes:

  Single-Object mode  (one item in frame)
    POST /reconstruct  →  rembg bg-removal → CLIP classify → LoRA route → TripoSR → Adaptive MC → GLB

  Crowded-Scene mode  (multiple items in frame)
    POST /segment      →  SAM click-to-select → returns cropped object + mask preview
    POST /reconstruct  →  (same as above, on the SAM-cropped image)

Startup loads ALL models once:
  - SAM (SamPredictor, sam_vit_b)
  - CLIP zero-shot classifier (openai/clip-vit-base-patch32)
  - TripoSR + LoRA injection + LoRAAdapterManager
  - rembg session

Endpoints:
  POST /segment          SAM segmentation with point or box prompt
  POST /reconstruct      CLIP classify → LoRA route → TripoSR → baked GLB
  GET  /static/models/*  Serve GLB files
  GET  /qr/{id}          QR code PNG pointing to /viewer/{id}
  GET  /viewer/{id}      model-viewer AR page
  POST /remove-bg        Background removal (kept for frontend compat)
  GET  /export           OBJ/GLB export (kept for frontend compat)

Run:
    python backend.py
"""

import base64
import io
import os
import socket
import time
import uuid
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FurnishAR Backend", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_MODELS_DIR = Path("static/models")
STATIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = Path(tempfile.gettempdir()) / "furnishar"
EXPORT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

sessions: dict = {}

# ── Category / adapter constants ──────────────────────────────────────────────
CLIP_LABELS = [
    "a bed", "a chair", "a sofa", "a swivel office chair", "a table",
    "a cabinet", "a bookshelf", "a desk", "a bench", "a wardrobe",
]

LABEL_TO_CATEGORY = {
    "a bed": "bed", "a chair": "chair", "a sofa": "sofa",
    "a swivel office chair": "swivelchair", "a table": "table",
    "a cabinet": "cabinet", "a bookshelf": "bookshelf",
    "a desk": "desk", "a bench": "bench", "a wardrobe": "wardrobe",
}

ADAPTER_MAP = {
    "chair":       "lora_chair_bench.pt",
    "bench":       "lora_chair_bench.pt",
    "table":       "lora_table_desk.pt",
    "desk":        "lora_table_desk.pt",
    "sofa":        "lora_sofa_stool.pt",
    "stool":       "lora_sofa_stool.pt",
    "swivelchair": "lora_sofa_stool.pt",
    "bed":         "lora_bed_wardrobe.pt",
    "wardrobe":    "lora_bed_wardrobe.pt",
    "cabinet":     "lora_cabinet_bookshelf.pt",
    "bookshelf":   "lora_cabinet_bookshelf.pt",
}

SAM_CHECKPOINT  = Path("checkpoints/sam_vit_b_01ec64.pth")
ADAPTER_DIR     = Path("adapters")

# ── Global model handles ──────────────────────────────────────────────────────
model          = None
adapter_manager = None
sam_predictor  = None
mobilenet_model = None
mobilenet_transform = None
rembg_session  = None
DEVICE         = "cpu"
TRIPOSR_READY  = False
SAM_READY      = False
CLIP_READY     = False
PUBLIC_URL     = None   # e.g. "https://abc123.ngrok-free.app"
USE_ADAPTIVE_MC = False  # set via --adaptive CLI flag
MC_RESOLUTION   = 256    # fine grid resolution, set via --mc-resolution
MC_COARSE_RESOLUTION = 128  # coarse grid resolution, set via --mc-coarse-resolution

# ── Startup — load everything once ───────────────────────────────────────────
def load_all_models():
    global model, adapter_manager, sam_predictor, mobilenet_model, mobilenet_transform
    global rembg_session, DEVICE, TRIPOSR_READY, SAM_READY, CLIP_READY

    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}", flush=True)

    # ── 1. rembg ──────────────────────────────────────────────────────────────
    try:
        import rembg
        rembg_session = rembg.new_session()
        print("rembg session ready.", flush=True)
    except Exception as e:
        print(f"rembg failed: {e}", flush=True)

    # ── 2. SAM ────────────────────────────────────────────────────────────────
    try:
        from segment_anything import SamPredictor, sam_model_registry
        if SAM_CHECKPOINT.exists():
            sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
            sam.to(DEVICE)
            sam_predictor = SamPredictor(sam)
            SAM_READY = True
            print("SAM ready.", flush=True)
        else:
            print(f"SAM checkpoint not found at {SAM_CHECKPOINT} — /segment disabled.", flush=True)
    except Exception as e:
        print(f"SAM load failed: {e}", flush=True)

    # ── 3. MobileNetV4 Lite ───────────────────────────────────────────────────
    try:
        import timm
        from torchvision import transforms
        mobilenet_model = timm.create_model('mobilenetv4_conv_small', pretrained=True)
        mobilenet_model.to(DEVICE)
        mobilenet_model.eval()

        mobilenet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        CLIP_READY = True
        print("MobileNetV4 Lite ready.", flush=True)
    except Exception as e:
        print(f"MobileNetV4 load failed: {e}", flush=True)

    # ── 4. TripoSR + LoRA ─────────────────────────────────────────────────────
    try:
        import torch
        from tsr.system import TSR
        from tsr.models.transformer.lora import LoRAAdapterManager

        model = TSR.from_pretrained(
            "./stabilityai",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        # Freeze base model and inject LoRA
        for param in model.parameters():
            param.requires_grad = False

        transformer = None
        for _, module in model.named_modules():
            if "Transformer1D" in type(module).__name__:
                transformer = module
                break
        if transformer is None:
            raise RuntimeError("Transformer1D not found in model")
        transformer.enable_lora(r=8, alpha=16)

        model.renderer.set_chunk_size(8192)
        model.to(DEVICE)
        model.eval()

        # Build adapter map from files that exist on disk
        adapter_file_map = {}
        for fname in ADAPTER_MAP.values():
            path = ADAPTER_DIR / fname
            if path.exists() and fname not in adapter_file_map:
                adapter_file_map[fname] = str(path)

        adapter_manager = LoRAAdapterManager(model, adapter_file_map)
        if adapter_file_map:
            adapter_manager.preload_all()
        print(f"TripoSR + LoRA ready. Adapters loaded: {list(adapter_file_map.keys())}", flush=True)

        TRIPOSR_READY = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"TripoSR load failed: {e}", flush=True)


print("=" * 55, flush=True)
print("  FurnishAR Backend v3 — loading models...", flush=True)
print("=" * 55, flush=True)
load_all_models()
print("=" * 55, flush=True)
print("  All models loaded. Starting server.", flush=True)
print("=" * 55, flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _base_url(port: int = 8000) -> str:
    """Return the public-facing base URL for links and QR codes."""
    if PUBLIC_URL:
        return PUBLIC_URL.rstrip("/")
    return f"http://{_local_ip()}:{port}"


def preprocess_image(img: Image.Image, do_remove_bg: bool = True) -> Image.Image:
    """Identical preprocessing to official run.py."""
    from tsr.utils import remove_background, resize_foreground
    img = img.convert("RGBA")
    if do_remove_bg and rembg_session is not None:
        img = remove_background(img, rembg_session)
    img = resize_foreground(img, 0.85)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr  = np.array(img).astype(np.float32) / 255.0
    rgb  = arr[:, :, :3]
    alpha = arr[:, :, 3:4]
    composited = rgb * alpha + 0.5 * (1.0 - alpha)
    return Image.fromarray((composited * 255.0).astype(np.uint8))


def _sam_segment(image_rgb: np.ndarray, point=None, box=None):
    """Run SAM with a point or box prompt. Returns (best_mask, best_score, all_scores)."""
    sam_predictor.set_image(image_rgb)
    point_coords = np.array([[point[0], point[1]]], dtype=np.float32) if point else None
    point_labels = np.array([1], dtype=np.int32) if point else None
    box_arr      = np.array(box, dtype=np.float32) if box else None
    masks, scores, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best], float(scores[best]), scores.tolist()


def _mask_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)


def _crop_on_white(image_rgb: np.ndarray, mask: np.ndarray, padding: float = 0.10):
    H, W = mask.shape
    x1, y1, x2, y2 = _mask_bbox(mask)
    pw = int((x2 - x1) * padding); ph = int((y2 - y1) * padding)
    cx1 = max(0, x1 - pw); cy1 = max(0, y1 - ph)
    cx2 = min(W, x2 + pw); cy2 = min(H, y2 + ph)
    crop = image_rgb[cy1:cy2, cx1:cx2].copy()
    cmask = mask[cy1:cy2, cx1:cx2]
    result = np.ones_like(crop) * 255
    result[cmask] = crop[cmask]
    # Make square
    h, w = result.shape[:2]
    side = max(h, w)
    square = np.ones((side, side, 3), dtype=np.uint8) * 255
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = result
    return Image.fromarray(square)


def _mask_preview(image_rgb: np.ndarray, mask: np.ndarray):
    from PIL import ImageDraw
    overlay = image_rgb.copy().astype(np.float32)
    overlay[~mask] = overlay[~mask] * 0.4
    overlay[mask]  = overlay[mask] * 0.6 + np.array([80, 220, 100]) * 0.4
    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = _mask_bbox(mask)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
    return img


def _img_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# Hardcoded ImageNet index mappings to our furniture categories
IMAGENET_FURNITURE_MAPPING = {
    "chair":       [423, 559, 765, 857],  # barber chair, folding chair, rocking chair, throne
    "bench":       [703],                 # park bench
    "table":       [532],                 # dining table
    "desk":        [514],                 # desk
    "sofa":        [805, 831],            # sofa, studio couch
    "stool":       [831],                 # fallback to studio couch/daybed
    "bed":         [823, 427, 512],      # four-poster bed, cradle, crib
    "wardrobe":    [894],                 # wardrobe
    "cabinet":     [490, 553, 646],      # china cabinet, file cabinet, medicine chest
    "bookshelf":   [456],                 # bookcase
    "swivelchair": [423],                 # fallback to chair index
}

def _classify(img: Image.Image):
    """Run MobileNetV4 Lite, return (category_str, confidence_float)."""
    import torch
    
    img_tensor = mobilenet_transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = mobilenet_model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        
    category_scores = {}
    for cat in IMAGENET_FURNITURE_MAPPING.keys():
        indices = IMAGENET_FURNITURE_MAPPING[cat]
        score = sum(probabilities[idx].item() for idx in indices)
        category_scores[cat] = score
        
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]
    
    if best_score < 0.01:
        return "chair", 0.0
        
    return best_category, best_score


def _set_adapter(category: str):
    """Hot-swap the LoRA adapter for the given category."""
    fname = ADAPTER_MAP.get(category)
    if fname and adapter_manager:
        try:
            adapter_manager.set_adapter(fname)
            return fname
        except Exception as e:
            print(f"Adapter swap failed ({fname}): {e}", flush=True)
    return None


def _normalize_category(category: str) -> str:
    """Clamp mobile/backend category input to a known LoRA route."""
    cleaned = (category or "").strip().lower().replace(" ", "")
    aliases = {
        "officechair": "swivelchair",
        "swivelofficechair": "swivelchair",
        "bookcase": "bookshelf",
    }
    cleaned = aliases.get(cleaned, cleaned)
    return cleaned if cleaned in ADAPTER_MAP else "chair"


def _apply_ar_upright_orientation(mesh):
    """Bake the reconstructed mesh into an upright, floor-aligned AR pose."""
    import trimesh

    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    bounds = np.asarray(mesh.bounds)
    if bounds.shape == (2, 3) and np.isfinite(bounds).all():
        center_x = float((bounds[0, 0] + bounds[1, 0]) * 0.5)
        center_z = float((bounds[0, 2] + bounds[1, 2]) * 0.5)
        min_y = float(bounds[0, 1])
        mesh.apply_translation([-center_x, -min_y, -center_z])
    return mesh


def _run_inference_baked(
    img: Image.Image,
    resolution: int = 256,
    texture_res: int = 2048,
    use_adaptive_mc=None,
):
    """
    Full pipeline:
      TripoSR encode →
        1. Extract mesh WITH vertex colors  → used for Three.js viewer JSON
           (uses adaptive MC if --adaptive flag is set, otherwise standard MC)
        2. Extract mesh WITHOUT vertex colors → bake texture → GLB for AR
      If texture baking fails, fall back to exporting the vertex-color mesh as GLB.

    Returns (glb_bytes, vc_mesh, scene_codes, elapsed_sec).
      - vc_mesh is the vertex-color mesh (higher quality for the Three.js viewer)
    """
    import torch
    import trimesh

    t0 = time.time()
    use_adaptive = USE_ADAPTIVE_MC if use_adaptive_mc is None else bool(use_adaptive_mc)

    amp_dtype = None
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        amp_dtype = torch.float16

    with torch.no_grad():
        if amp_dtype:
            with torch.autocast("cuda", dtype=amp_dtype):
                scene_codes = model([img], device=DEVICE)
        else:
            scene_codes = model([img], device=DEVICE)
    scene_codes = scene_codes.float()

    # ── 1. Vertex-color mesh ──────────────────────────────────────────────
    if use_adaptive:
        try:
            from adaptive_mc import extract_mesh_adaptive
            vc_mesh = extract_mesh_adaptive(
                model,
                scene_codes[0],
                device=DEVICE,
                resolution=resolution,
                coarse_resolution=MC_COARSE_RESOLUTION,
                has_vertex_color=True,
            )
            print(f"Adaptive MC (vertex-color): {len(vc_mesh.vertices):,} verts, {len(vc_mesh.faces):,} faces", flush=True)
        except Exception as e:
            print(f"Adaptive MC failed ({e}) — falling back to standard.", flush=True)
            vc_meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
            vc_mesh = vc_meshes[0]
    else:
        vc_meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
        vc_mesh = vc_meshes[0]
        print(f"Standard MC (vertex-color): {len(vc_mesh.vertices):,} verts, {len(vc_mesh.faces):,} faces", flush=True)
    _apply_ar_upright_orientation(vc_mesh)

    # ── 2. Try texture-baked GLB for model-viewer / AR
    glb_bytes = None
    try:
        import xatlas
        from tsr.bake_texture import bake_texture

        # Extract a second mesh WITHOUT vertex colors — bake_texture needs raw geometry
        if use_adaptive:
            try:
                from adaptive_mc import extract_mesh_adaptive
                raw_mesh = extract_mesh_adaptive(
                    model,
                    scene_codes[0],
                    device=DEVICE,
                    resolution=resolution,
                    coarse_resolution=MC_COARSE_RESOLUTION,
                    has_vertex_color=False,
                )
                print(f"Adaptive MC (raw): {len(raw_mesh.vertices):,} verts, {len(raw_mesh.faces):,} faces", flush=True)
            except Exception as e:
                print(f"Adaptive MC raw mesh failed ({e}) — falling back to standard.", flush=True)
                raw_meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=resolution)
                raw_mesh = raw_meshes[0]
        else:
            raw_meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=resolution)
            raw_mesh = raw_meshes[0]

        bake_output = bake_texture(raw_mesh, model, scene_codes[0], texture_res)
        texture_img = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)

        # xatlas UV unwrap → trimesh with embedded texture → GLB bytes
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "mesh.obj")
            xatlas.export(
                obj_path,
                raw_mesh.vertices[bake_output["vmapping"]],
                bake_output["indices"],
                bake_output["uvs"],
                raw_mesh.vertex_normals[bake_output["vmapping"]],
            )
            material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
            vis = trimesh.visual.TextureVisuals(uv=bake_output["uvs"], material=material)
            textured = trimesh.Trimesh(
                vertices=raw_mesh.vertices[bake_output["vmapping"]],
                faces=bake_output["indices"],
                vertex_normals=raw_mesh.vertex_normals[bake_output["vmapping"]],
                visual=vis,
                process=False,
            )
            _apply_ar_upright_orientation(textured)
            buf = io.BytesIO()
            textured.export(buf, file_type="glb")
            glb_bytes = buf.getvalue()
        print("Texture bake succeeded — GLB has embedded texture.", flush=True)
    except Exception as e:
        print(f"Texture bake failed ({e}) — falling back to vertex-color GLB.", flush=True)

    # ── Fallback: export vertex-color mesh as GLB directly
    if glb_bytes is None:
        buf = io.BytesIO()
        vc_mesh.export(buf, file_type="glb")
        glb_bytes = buf.getvalue()

    elapsed = round(time.time() - t0, 2)
    return glb_bytes, vc_mesh, scene_codes, elapsed


def _store_reconstruction_result(
    glb_bytes: bytes,
    vc_mesh,
    detected_category: str,
    detected_confidence: float,
    elapsed: float,
    mobile_metadata=None,
):
    session_id = str(uuid.uuid4())
    glb_path = STATIC_MODELS_DIR / f"{session_id}.glb"
    glb_path.write_bytes(glb_bytes)

    v = np.array(vc_mesh.vertices)
    f = np.array(vc_mesh.faces)

    vertex_colors = None
    try:
        if hasattr(vc_mesh, "visual") and hasattr(vc_mesh.visual, "vertex_colors"):
            vc = np.array(vc_mesh.visual.vertex_colors)
            vertex_colors = (vc[:, :3].astype(np.float32) / 255.0).tolist()
            print(f"Vertex colors extracted: {len(vertex_colors)} vertices", flush=True)
    except Exception as e:
        print(f"Could not extract vertex colors: {e}", flush=True)

    sessions[session_id] = {
        "vertices": v,
        "faces": f,
        "mesh_obj": vc_mesh,
        "glb_path": str(glb_path),
        "category": detected_category,
        "confidence": detected_confidence,
        "created": time.time(),
    }
    if mobile_metadata:
        sessions[session_id]["mobile"] = mobile_metadata

    if len(sessions) > 20:
        oldest = sorted(sessions, key=lambda k: sessions[k]["created"])[:5]
        for k in oldest:
            old_glb = Path(sessions[k].get("glb_path", ""))
            if old_glb.exists():
                old_glb.unlink(missing_ok=True)
            del sessions[k]

    resp = {
        "session_id": session_id,
        "glb_url": f"/static/models/{session_id}.glb",
        "viewer_url": f"/viewer/{session_id}",
        "category": detected_category,
        "confidence": round(float(detected_confidence), 4),
        "n_vertices": int(len(v)),
        "n_faces": int(len(f)),
        "time_sec": elapsed,
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "f_score": f"{detected_confidence:.0%}",
        "demo": False,
    }
    if vertex_colors is not None:
        resp["vertex_colors"] = vertex_colors
    if mobile_metadata:
        resp["mobile"] = mobile_metadata
    return JSONResponse(resp)


@app.post("/mobile/reconstruct")
async def mobile_reconstruct(
    file: UploadFile = File(...),
    category: str = Form(...),
    confidence: float = Form(0.0),
    segmentation: bool = Form(False),
    resolution: int = Form(None),
    texture_res: int = Form(2048),
):
    """Use phone-side ONNX preprocessing/classification, then reconstruct."""
    if not TRIPOSR_READY:
        raise HTTPException(503, "TripoSR model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image, got: {file.content_type}")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    resolution = max(64, min(512, int(resolution if resolution is not None else MC_RESOLUTION)))
    texture_res = max(512, min(4096, int(texture_res)))
    detected_category = _normalize_category(category)
    detected_confidence = max(0.0, min(1.0, float(confidence)))

    active_adapter = _set_adapter(detected_category)
    print(
        f"Mobile category: {detected_category} ({detected_confidence:.1%}) | "
        f"Adapter: {active_adapter} | Standard MC",
        flush=True,
    )

    try:
        img = preprocess_image(img, do_remove_bg=False)
        glb_bytes, vc_mesh, scene_codes, elapsed = _run_inference_baked(
            img,
            resolution=resolution,
            texture_res=texture_res,
            use_adaptive_mc=False,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Mobile reconstruction failed: {e}")

    return _store_reconstruction_result(
        glb_bytes,
        vc_mesh,
        detected_category,
        detected_confidence,
        elapsed,
        mobile_metadata={
            "segmentation": bool(segmentation),
            "adapter": active_adapter,
            "mc_mode": "standard",
            "client_preprocessed": True,
        },
    )


# ── Basic endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "FurnishAR.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>FurnishAR.html not found</h1>", status_code=404)

@app.get("/health")
def health():
    modes = ["single_object"]
    if SAM_READY:
        modes.append("crowded_scene")
    return {
        "status": "running",
        "triposr_ready": TRIPOSR_READY,
        "sam_ready": SAM_READY,
        "clip_ready": CLIP_READY,
        "device": DEVICE,
        "pipeline_modes": modes,
    }



# ── POST /segment ─────────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(
    file:    UploadFile = File(...),
    point_x: float = Form(None),
    point_y: float = Form(None),
    box_x1:  float = Form(None),
    box_y1:  float = Form(None),
    box_x2:  float = Form(None),
    box_y2:  float = Form(None),
):
    if not SAM_READY:
        raise HTTPException(503, "SAM model not loaded")

    has_point = point_x is not None and point_y is not None
    has_box   = all(v is not None for v in [box_x1, box_y1, box_x2, box_y2])
    if not has_point and not has_box:
        raise HTTPException(400, "Provide either (point_x, point_y) or (box_x1, box_y1, box_x2, box_y2)")

    raw = await file.read()
    try:
        pil_img   = Image.open(io.BytesIO(raw)).convert("RGB")
        image_rgb = np.array(pil_img)
    except Exception:
        raise HTTPException(400, "Could not decode image")

    try:
        point = (int(point_x), int(point_y)) if has_point else None
        box   = (int(box_x1), int(box_y1), int(box_x2), int(box_y2)) if has_box else None
        mask, score, all_scores = _sam_segment(image_rgb, point=point, box=box)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"SAM segmentation failed: {e}")

    x1, y1, x2, y2 = _mask_bbox(mask)
    area_px = int(mask.sum())
    H, W = image_rgb.shape[:2]

    cropped = _crop_on_white(image_rgb, mask)
    preview = _mask_preview(image_rgb, mask)

    return JSONResponse({
        "mask_confidence":  round(score, 4),
        "all_scores":       [round(s, 4) for s in all_scores],
        "bbox":             [x1, y1, x2, y2],
        "area_px":          area_px,
        "area_pct":         round(100 * area_px / (H * W), 2),
        "cropped_image":    _img_to_b64(cropped),
        "mask_preview":     _img_to_b64(preview),
    })


# ── POST /classify ────────────────────────────────────────────────────────────

@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    if not CLIP_READY:
        raise HTTPException(503, "Classifier model not loaded")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not decode image")
    try:
        img_pre = preprocess_image(img, do_remove_bg=True)
        detected_category, detected_confidence = _classify(img_pre)
        return JSONResponse({
            "category": detected_category,
            "confidence": float(detected_confidence)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Classification failed: {e}")


# ── POST /reconstruct ─────────────────────────────────────────────────────────

@app.post("/reconstruct")
async def reconstruct(
    file:        UploadFile = File(...),
    category:    str  = Form("auto"),   # kept for backward compat; ignored (CLIP classifies)
    resolution:  int  = Form(None),
    remove_bg:   bool = Form(True),
    texture_res: int  = Form(2048),
):
    if not TRIPOSR_READY:
        raise HTTPException(503, "TripoSR model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image, got: {file.content_type}")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not decode image")

    resolution  = max(64, min(512, int(resolution if resolution is not None else MC_RESOLUTION)))
    texture_res = max(512, min(4096, int(texture_res)))

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        img = preprocess_image(img, do_remove_bg=remove_bg)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # ── Classify ──────────────────────────────────────────────────────────────
    detected_category  = "chair"
    detected_confidence = 0.0
    if CLIP_READY:
        try:
            detected_category, detected_confidence = _classify(img)
        except Exception as e:
            print(f"CLIP classify failed: {e} — defaulting to 'chair'", flush=True)
    else:
        # Fall back to the manually passed category
        detected_category = category if category != "auto" else "chair"

    # ── Route adapter ─────────────────────────────────────────────────────────
    active_adapter = _set_adapter(detected_category)
    print(f"Category: {detected_category} ({detected_confidence:.1%}) | Adapter: {active_adapter}", flush=True)

    # ── Inference + texture bake ──────────────────────────────────────────────
    try:
        glb_bytes, vc_mesh, scene_codes, elapsed = _run_inference_baked(
            img, resolution=resolution, texture_res=texture_res
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Reconstruction failed: {e}")

    # ── Save GLB ──────────────────────────────────────────────────────────────
    return _store_reconstruction_result(
        glb_bytes,
        vc_mesh,
        detected_category,
        detected_confidence,
        elapsed,
    )


# ── GET /qr/{session_id} ──────────────────────────────────────────────────────

@app.get("/qr/{session_id}")
def get_qr(session_id: str, port: int = 8000):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    try:
        import qrcode
    except ImportError:
        raise HTTPException(500, "Install qrcode: pip install qrcode[pil]")

    base = _base_url(port)
    viewer_url = f"{base}/viewer/{session_id}"

    qr = qrcode.QRCode(box_size=8, border=4)
    qr.add_data(viewer_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# ── GET /viewer/{session_id} ──────────────────────────────────────────────────

@app.get("/viewer/{session_id}", response_class=HTMLResponse)
def viewer(session_id: str, port: int = 8000):
    session = sessions.get(session_id)
    if not session:
        return HTMLResponse("<h2>Session not found or expired.</h2>", status_code=404)

    base = _base_url(port)
    glb_url     = f"{base}/static/models/{session_id}.glb"
    qr_url      = f"{base}/qr/{session_id}?port={port}"
    category    = session.get("category", "object")
    confidence  = session.get("confidence", 0.0)
    n_vertices  = int(len(session["vertices"]))
    n_faces     = int(len(session["faces"]))

    # Read viewer template if it exists, otherwise use inline template
    viewer_tmpl = TEMPLATES_DIR / "viewer.html"
    if viewer_tmpl.exists():
        html = viewer_tmpl.read_text(encoding="utf-8")
        for key, val in [
            ("{GLB_URL}", glb_url),
            ("{QR_URL}", qr_url),
            ("{CATEGORY}", category.capitalize()),
            ("{CONFIDENCE}", f"{confidence:.1%}"),
            ("{N_VERTICES}", f"{n_vertices:,}"),
            ("{N_FACES}", f"{n_faces:,}"),
            ("{SESSION_ID}", session_id),
        ]:
            html = html.replace(key, str(val))
        return HTMLResponse(html)

    # Inline fallback
    return HTMLResponse(_viewer_html(
        glb_url, qr_url, category, confidence, n_vertices, n_faces, session_id
    ))


def _viewer_html(glb_url, qr_url, category, confidence, n_vertices, n_faces, session_id):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FurnishAR Viewer</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0f0f13;color:#e8e8f0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center}}
    header{{width:100%;padding:16px 24px;background:#1a1a24;border-bottom:1px solid #2a2a3a;display:flex;align-items:center;gap:12px}}
    header h1{{font-size:1.3rem;font-weight:700;color:#a78bfa}}
    header span{{font-size:.85rem;color:#888;margin-left:auto}}
    .viewer-wrap{{width:100%;max-width:900px;flex:1;padding:20px}}
    model-viewer{{width:100%;height:500px;background:#1a1a24;border-radius:16px;border:1px solid #2a2a3a;--poster-color:#1a1a24}}
    .meta{{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}}
    .chip{{background:#1e1e2e;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-size:.85rem}}
    .chip strong{{color:#a78bfa}}
    .qr-section{{width:100%;max-width:900px;padding:0 20px 24px;display:flex;flex-direction:column;align-items:center;gap:12px}}
    .qr-section h2{{font-size:1rem;color:#888}}
    .qr-section img{{width:180px;height:180px;border-radius:12px;border:3px solid #a78bfa;background:#fff;padding:4px}}
    .qr-hint{{font-size:.8rem;color:#666;text-align:center}}
    @media(max-width:600px){{model-viewer{{height:380px}}}}
  </style>
</head>
<body>
  <header>
    <h1>FurnishAR</h1>
    <span>3D Furniture Reconstruction</span>
  </header>

  <div class="viewer-wrap">
    <model-viewer
      src="{glb_url}"
      ar
      ar-modes="webxr scene-viewer quick-look"
      camera-controls
      auto-rotate
      shadow-intensity="1"
      exposure="1"
      style="--progress-bar-color:#a78bfa"
      alt="Reconstructed {category}">
      <button slot="ar-button" style="
        position:absolute;bottom:16px;right:16px;
        background:#a78bfa;color:#fff;border:none;
        border-radius:24px;padding:10px 22px;
        font-size:1rem;font-weight:600;cursor:pointer">
        &#x1F4F1; View in AR
      </button>
    </model-viewer>

    <div class="meta">
      <div class="chip">Category: <strong>{category.capitalize()}</strong></div>
      <div class="chip">Confidence: <strong>{confidence:.1%}</strong></div>
      <div class="chip">Vertices: <strong>{n_vertices:,}</strong></div>
      <div class="chip">Faces: <strong>{n_faces:,}</strong></div>
    </div>
  </div>

  <div class="qr-section">
    <h2>Scan to view AR on your phone</h2>
    <img src="{qr_url}" alt="QR code"/>
    <p class="qr-hint">iOS: tap &amp; hold → Open in AR<br>Android: Chrome → AR button</p>
  </div>
</body>
</html>"""


# ── POST /set-url — update public URL at runtime ─────────────────────────────

@app.post("/set-url")
async def set_public_url(url: str = Form(...)):
    global PUBLIC_URL
    url = url.strip()
    PUBLIC_URL = url if url else None
    print(f"Public URL updated → {PUBLIC_URL or '(cleared, using local IP)'}", flush=True)
    return JSONResponse({"public_url": PUBLIC_URL, "base_url": _base_url()})


# ── POST /remove-bg (kept for frontend compat) ────────────────────────────────

@app.post("/remove-bg")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    if rembg_session is None:
        raise HTTPException(503, "rembg not loaded")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    try:
        from tsr.utils import remove_background
        img = remove_background(img, rembg_session)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Background removal failed: {e}")


# ── GET /export (kept for frontend compat) ────────────────────────────────────

@app.get("/export")
def export_mesh(session_id: str, format: str = "obj"):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if format == "glb":
        glb_path = session.get("glb_path")
        if glb_path and Path(glb_path).exists():
            return FileResponse(glb_path, media_type="model/gltf-binary",
                                filename="furnishar_model.glb")
        raise HTTPException(404, "GLB not found for this session")

    if format == "obj":
        vertices = session["vertices"]
        faces    = session["faces"]
        lines    = [f"# FurnishAR {len(vertices)} vertices / {len(faces)} faces", ""]
        for v in vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        lines.append("")
        for f in faces:
            lines.append(f"f {int(f[0])+1} {int(f[1])+1} {int(f[2])+1}")
        path = EXPORT_DIR / f"{session_id}.obj"
        path.write_text("\n".join(lines))
        return FileResponse(path, media_type="text/plain", filename="furnishar_model.obj")

    raise HTTPException(400, f"Unknown format '{format}'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="FurnishAR Backend")
    parser.add_argument("--public-url", default=None,
                        help="Public base URL for QR codes and viewer links "
                             "(e.g. https://abc123.ngrok-free.app)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--adaptive", action="store_true", default=False,
                        help="Use adaptive marching cubes for faster mesh extraction "
                             "(coarse-to-fine, ~2-4x speedup). Without this flag, "
                             "standard full-grid marching cubes is used.")
    parser.add_argument("--mc-resolution", type=int, default=256,
                        help="Marching cubes fine grid resolution (default: 256). "
                             "Higher = more detail but slower. Clamped to [64, 512].")
    parser.add_argument("--mc-coarse-resolution", type=int, default=128,
                        help="Coarse grid resolution for adaptive MC (default: 128). "
                             "Only used with --adaptive. Should be < mc-resolution.")
    args = parser.parse_args()

    if args.public_url:
        PUBLIC_URL = args.public_url.rstrip("/")
    USE_ADAPTIVE_MC = args.adaptive
    MC_RESOLUTION = max(64, min(512, args.mc_resolution))
    MC_COARSE_RESOLUTION = max(32, min(MC_RESOLUTION, args.mc_coarse_resolution))

    if USE_ADAPTIVE_MC:
        mc_mode = f"Adaptive (fine={MC_RESOLUTION}, coarse={MC_COARSE_RESOLUTION})"
    else:
        mc_mode = f"Standard (full-grid {MC_RESOLUTION}\u00b3)"
    ip = _local_ip()
    print(f"\n  Server:   http://0.0.0.0:{args.port}")
    print(f"  Local:    http://localhost:{args.port}")
    print(f"  Network:  http://{ip}:{args.port}")
    print(f"  MC mode:  {mc_mode}")
    if PUBLIC_URL:
        print(f"  Public:   {PUBLIC_URL}")
    print(f"  Frontend: open FurnishAR.html\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
