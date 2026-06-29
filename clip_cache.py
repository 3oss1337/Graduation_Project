"""Local CLIP model cache for FurnishAR classification."""

from pathlib import Path
import os


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_cache_dir() -> Path:
    raw_dir = os.environ.get("FURNISHAR_CLIP_DIR", "models/clip-vit-base-patch32")
    path = Path(raw_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _has_saved_clip_model(cache_dir: Path) -> bool:
    model_files = (
        cache_dir / "model.safetensors",
        cache_dir / "pytorch_model.bin",
    )
    processor_files = (
        cache_dir / "processor_config.json",
        cache_dir / "preprocessor_config.json",
    )
    tokenizer_files = (
        cache_dir / "tokenizer.json",
        cache_dir / "vocab.json",
    )
    required_files = (
        cache_dir / "config.json",
        cache_dir / "tokenizer_config.json",
    )
    return (
        any(path.exists() for path in model_files)
        and any(path.exists() for path in processor_files)
        and any(path.exists() for path in tokenizer_files)
        and all(path.exists() for path in required_files)
    )


def get_clip_model_path() -> str:
    """Return a local CLIP path, downloading and saving it once if needed."""
    model_id = os.environ.get("FURNISHAR_CLIP_MODEL", DEFAULT_CLIP_MODEL_ID)
    cache_dir = _configured_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not _has_saved_clip_model(cache_dir):
        from transformers import CLIPModel, CLIPProcessor

        print(f"Saving CLIP model locally to {cache_dir} from {model_id}...", flush=True)
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.save_pretrained(cache_dir)
        processor.save_pretrained(cache_dir)

    return str(cache_dir)
