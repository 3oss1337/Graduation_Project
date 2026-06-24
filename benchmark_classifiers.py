import time
import torch
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from transformers import pipeline as hf_pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "examples + 3D lora output/Chair1.png"
NUM_WARMUP = 10
NUM_RUNS = 50

CATEGORIES = [
    "chair", "table", "sofa", "bed", "cabinet",
    "bookshelf", "desk", "bench", "swivelchair", "wardrobe",
]

IMAGENET_FURNITURE_MAPPING = {
    "chair":       [423, 559, 765, 857],
    "bench":       [703],
    "table":       [532],
    "desk":        [514],
    "sofa":        [805, 831],
    "stool":       [831],
    "bed":         [823, 427, 512],
    "wardrobe":    [894],
    "cabinet":     [490, 553, 646],
    "bookshelf":   [456],
    "swivelchair": [423],
}

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # Fallback to random image if not found
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

def benchmark_clip(image, device):
    print(f"\n[CLIP] Loading Model (openai/clip-vit-base-patch32)...")
    clip_pipeline = hf_pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=0 if device == "cuda" else -1,
    )
    
    # Warmup
    print(f"[CLIP] Performing {NUM_WARMUP} warmup iterations...")
    for _ in range(NUM_WARMUP):
        _ = clip_pipeline(image, candidate_labels=CATEGORIES)
        
    if device == "cuda":
        torch.cuda.synchronize()
        
    print(f"[CLIP] Running {NUM_RUNS} benchmark runs...")
    latencies = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        _ = clip_pipeline(image, candidate_labels=CATEGORIES)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000) # ms
        
    return latencies

def benchmark_mobilenet(image, device):
    print(f"\n[MobileNetV4] Loading Model (mobilenetv4_conv_small)...")
    model = timm.create_model('mobilenetv4_conv_small', pretrained=True)
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess image once
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Warmup
    print(f"[MobileNetV4] Performing {NUM_WARMUP} warmup iterations...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            all_scores = {}
            for cat in CATEGORIES:
                indices = IMAGENET_FURNITURE_MAPPING.get(cat, [])
                score = sum(probabilities[idx].item() for idx in indices)
                all_scores[cat] = score
                
    if device == "cuda":
        torch.cuda.synchronize()
        
    print(f"[MobileNetV4] Running {NUM_RUNS} benchmark runs...")
    latencies = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            all_scores = {}
            for cat in CATEGORIES:
                indices = IMAGENET_FURNITURE_MAPPING.get(cat, [])
                score = sum(probabilities[idx].item() for idx in indices)
                all_scores[cat] = score
                
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000) # ms
        
    return latencies

def print_stats(name, latencies):
    mean_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    std_lat = np.std(latencies)
    throughput = 1000.0 / mean_lat
    
    print(f"\n==================================================")
    print(f"  {name} RESULTS")
    print(f"==================================================")
    print(f"  Average Latency: {mean_lat:.3f} ms")
    print(f"  Min Latency:     {min_lat:.3f} ms")
    print(f"  Max Latency:     {max_lat:.3f} ms")
    print(f"  Std Dev:         {std_lat:.3f} ms")
    print(f"  Throughput:      {throughput:.1f} images/sec")
    print(f"==================================================")
    
    return {
        "mean": mean_lat,
        "min": min_lat,
        "max": max_lat,
        "std": std_lat,
        "throughput": throughput
    }

def main():
    print("=" * 50)
    print("  CLASSIFICATION SPEED BENCHMARK")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print(f"  Image:  {IMAGE_PATH}")
    print(f"  Runs:   {NUM_RUNS} iterations ({NUM_WARMUP} warmup)")
    print("=" * 50)
    
    image = load_image(IMAGE_PATH)
    
    # 1. Benchmark CLIP
    clip_lats = benchmark_clip(image, DEVICE)
    clip_stats = print_stats("CLIP ZERO-SHOT", clip_lats)
    
    # 2. Benchmark MobileNetV4
    mn_lats = benchmark_mobilenet(image, DEVICE)
    mn_stats = print_stats("MOBILENETV4 CONV SMALL", mn_lats)
    
    # Summary Table
    print("\n" + "=" * 65)
    print(f"  {'Model':<25} | {'Avg Latency':<12} | {'Throughput':<15}")
    print("-" * 65)
    print(f"  {'CLIP Zero-Shot':<25} | {clip_stats['mean']:>8.2f} ms  | {clip_stats['throughput']:>10.1f} img/s")
    print(f"  {'MobileNetV4 Conv Small':<25} | {mn_stats['mean']:>8.2f} ms  | {mn_stats['throughput']:>10.1f} img/s")
    print("=" * 65)
    
    speedup = clip_stats['mean'] / mn_stats['mean']
    print(f"  >> MobileNetV4 is {speedup:.1f}x FASTER than CLIP zero-shot!\n")

if __name__ == "__main__":
    main()
