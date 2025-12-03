"""
Pretrained Deformable DETR wrapper using HuggingFace Transformers
=================================================================

This file provides:
    - inference(config, checkpoint_path, image_path, save_path)
with the SAME signature as used by your inference_detr_example.py.

It loads a pretrained Deformable DETR model from HuggingFace:
    "SenseTime/deformable-detr"

NOTE:
- This model is trained on COCO classes (91 categories).
- It does NOT know your urban issue labels unless you fine-tune it.
"""

import os
from typing import Tuple, List

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor


# ---------------------------------------------------------------------------
# Global model cache (so we don't reload for every image)
# ---------------------------------------------------------------------------
_MODEL_NAME = "SenseTime/deformable-detr"

_processor = None
_model = None
_device = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_model(checkpoint_path: str = None):
    """
    Load the pretrained Deformable DETR model + processor.

    If `checkpoint_path` is a directory or local HF checkpoint, it will try to load from there.
    Otherwise it defaults to the HuggingFace model "SenseTime/deformable-detr".
    """
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    device = _get_device()

    # Decide source (remote HF vs local fine-tuned)
    src = checkpoint_path if (checkpoint_path and os.path.exists(checkpoint_path)) else _MODEL_NAME

    print(f"[Deformable DETR] Loading model from: {src}")
    _processor = DeformableDetrImageProcessor.from_pretrained(src)
    _model = DeformableDetrForObjectDetection.from_pretrained(src).to(device)
    _model.eval()

    return _processor, _model


# ---------------------------------------------------------------------------
# Helper: visualization
# ---------------------------------------------------------------------------
def _draw_boxes_on_image(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    id2label: dict,
    score_threshold: float = 0.5
) -> Image.Image:
    draw = ImageDraw.Draw(image)

    # Try to load a default font (optional)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, label_id, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x0, y0, x1, y1 = box
        # Box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        # Label text
        label_name = id2label.get(label_id, str(label_id))
        text = f"{label_name}: {score:.2f}"

        if font is not None:
            text_size = draw.textsize(text, font=font)
        else:
            text_size = draw.textsize(text)

        text_bg = [x0, y0 - text_size[1], x0 + text_size[0], y0]
        draw.rectangle(text_bg, fill="red")
        if font is not None:
            draw.text((x0, y0 - text_size[1]), text, fill="white", font=font)
        else:
            draw.text((x0, y0 - text_size[1]), text, fill="white")

    return image


# ---------------------------------------------------------------------------
# Main API: inference()
# ---------------------------------------------------------------------------
def inference(config, checkpoint_path, image_path: str, save_path: str = None):
    """
    Run inference on a single image using pretrained Deformable DETR.

    Parameters
    ----------
    config : Config
        Your existing Config object. We mainly read:
            - config.conf_threshold (if present) as confidence threshold
    checkpoint_path : str
        Ignored for HuggingFace pretrained (unless it's a local HF checkpoint directory).
        Kept for API compatibility with your script.
    image_path : str
        Path to the input image.
    save_path : str
        Path to save the output image with boxes drawn. If None, no image is saved.

    Returns
    -------
    boxes : List[List[float]]
        Bounding boxes in [x0, y0, x1, y1] format.
    labels : List[int]
        COCO class IDs (not your 0-9 urban labels).
    scores : List[float]
        Confidence scores between 0 and 1.
    """
    # Load model & processor (pretrained)
    processor, model = _load_model(checkpoint_path)

    device = _get_device()

    # Confidence threshold from config or default 0.5
    score_threshold = getattr(config, "conf_threshold", 0.5)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get boxes in original image size
    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (height, width)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=score_threshold
    )[0]

    boxes = results["boxes"].cpu().tolist()      # [N, 4]
    scores = results["scores"].cpu().tolist()    # [N]
    labels = results["labels"].cpu().tolist()    # [N]

    # Optionally draw and save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis_img = image.copy()
        vis_img = _draw_boxes_on_image(
            vis_img,
            boxes,
            labels,
            scores,
            id2label=model.config.id2label,
            score_threshold=score_threshold
        )
        vis_img.save(save_path)

    return boxes, labels, scores


# ---------------------------------------------------------------------------
# Optional: dummy test() so your testing script doesn't crash
# ---------------------------------------------------------------------------
def test(config, checkpoint_path=None):
    """
    Dummy test function for interface compatibility with your test script.

    It does NOT compute real mAP/precision/recall for your urban classes,
    because the pretrained model is on COCO.

    It just returns a dict with placeholder metrics, so your example test
    script can run without error.

    If you really want evaluation, you must fine-tune the model on your
    dataset and implement proper mapping between predictions and labels.
    """
    print("[Deformable DETR] WARNING: `test()` with pretrained COCO model is only a placeholder.")
    print("It does not compute real mAP/precision/recall for your urban issue classes.\n")

    # You could optionally run some sanity-check inference on a few test images here.

    results = {
        "mAP@0.5": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    return results


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class DummyConfig:
        conf_threshold = 0.5
        output_dir = "outputs_deformable_detr_pretrained"

    cfg = DummyConfig()
    img_path = r'C:\Users\super\Downloads\pot_homejpg.jpg'
    out_path = os.path.join(cfg.output_dir, "test_output.jpg")
    boxes, labels, scores = inference(cfg, None, img_path, out_path)
    print(f"Detected {len(boxes)} objects.")
