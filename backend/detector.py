import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DIRECTML_DISABLE'] = '1'

import tensorflow as tf
import numpy as np
from PIL import Image

BINARY_MODEL_PATH     = r"C:\Workspace\CNN\backend\models\binary_cnn"
MULTICLASS_MODEL_PATH = r"C:\Workspace\CNN\backend\models\multiclass_cnn"
RCNN_MODEL_PATH       = r"C:\Workspace\CNN\faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model"

BINARY_CLASSES     = {0: "airplane", 1: "car"}
MULTICLASS_CLASSES = {0: "airplane", 1: "car", 8: "boat"}
RCNN_CLASSES       = {3: "car", 5: "airplane", 9: "boat"}

print("Loading models...")
binary_model     = tf.keras.models.load_model(BINARY_MODEL_PATH)
multiclass_model = tf.keras.models.load_model(MULTICLASS_MODEL_PATH)
rcnn_model       = tf.saved_model.load(RCNN_MODEL_PATH)
print("All models loaded.")

def predict_binary(image: Image.Image):
    img = image.resize((32, 32)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    score = float(binary_model.predict(arr)[0][0])
    label = "car" if score > 0.5 else "airplane"
    confidence = score if score > 0.5 else 1 - score
    return {"label": label, "confidence": round(confidence, 4)}

def predict_multiclass(image: Image.Image):
    img = image.resize((32, 32)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = multiclass_model.predict(arr)[0]
    class_indices = {0: "airplane", 1: "car", 8: "boat"}
    mapped = {class_indices[i]: float(probs[i]) for i in class_indices}
    label = max(mapped, key=mapped.get)
    return {"label": label, "confidence": round(mapped[label], 4), "all_scores": mapped}

def predict_rcnn(image: Image.Image):
    img = image.convert("RGB")
    arr = np.array(img)
    tensor = tf.convert_to_tensor(arr)[tf.newaxis, ...]
    detections = rcnn_model(tensor)
    boxes   = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)
    scores  = detections["detection_scores"][0].numpy()
    results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score < 0.5 or cls not in RCNN_CLASSES:
            continue
        ymin, xmin, ymax, xmax = [round(float(c), 4) for c in box]
        results.append({
            "label": RCNN_CLASSES[cls],
            "confidence": round(float(score), 4),
            "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        })
    return results
