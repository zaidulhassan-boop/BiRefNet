# handler.py
import base64, io, os, re, requests
import numpy as np
from PIL import Image
import torch

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/models/birefnet")
_model, _processor = None, None


def _load_model():
    """Load BiRefNet model once (from cached local directory)."""
    global _model, _processor
    if _model is not None:
        return

    from transformers import AutoModelForImageSegmentation, AutoImageProcessor
    _processor = AutoImageProcessor.from_pretrained(
        MODEL_LOCAL_DIR, trust_remote_code=True
    )
    _model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_LOCAL_DIR, trust_remote_code=True
    ).to(DEVICE).eval()


def _fetch_image(image_b64=None, image_url=None) -> Image.Image:
    """Get image from base64 or URL."""
    if image_b64:
        raw = base64.b64decode(re.sub("^data:image/[^;]+;base64,", "", image_b64))
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if image_url:
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise ValueError("No valid image input provided.")


def _predict_mask(pil_img: Image.Image) -> np.ndarray:
    """Generate alpha mask (0–1 float array)."""
    _load_model()
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    inputs = _processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = _model(**inputs)
        pred = (
            outputs.get("predictions")
            or outputs.get("logits")
            or list(outputs.values())[0]
        )
        if pred.dim() == 4:
            pred = torch.nn.functional.interpolate(
                pred, size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        mask = torch.sigmoid(pred).cpu().numpy()
        return mask


def remove_bg_to_png_b64(pil_img: Image.Image, threshold=0.9) -> str:
    """Apply mask to make background transparent, return base64 PNG."""
    mask = _predict_mask(pil_img)
    alpha = ((mask >= threshold) * 255).astype(np.uint8)
    rgba = Image.new("RGBA", pil_img.size)
    rgba.paste(pil_img, (0, 0))
    rgba.putalpha(Image.fromarray(alpha))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def rp_handler(event):
    """
    RunPod handler.
    Input JSON:
    {
      "image_url": "https://...",
      or
      "image_b64": "...",
      "threshold": 0.9
    }
    """
    data = event.get("input", event)
    img = _fetch_image(data.get("image_b64"), data.get("image_url"))
    out_b64 = remove_bg_to_png_b64(img, data.get("threshold", 0.9))
    return {"ok": True, "png_b64": out_b64}
