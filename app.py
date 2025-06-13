import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import joblib
from rembg import remove
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model
yolo_model = YOLO(os.path.join(BASE_DIR, "best.pt"))

# Load ONNX MobileNetV2 plant type classifier
onnx_session = ort.InferenceSession(os.path.join(BASE_DIR, "mobilenetv2_plant_classifier.onnx"))
input_name = onnx_session.get_inputs()[0].name

# Class names
class_names = ['apple', 'cherry', 'grapes', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']

def resize_with_background(image, size=(224, 224), background_color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_pil.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', size, background_color)
    x = (size[0] - image_pil.width) // 2
    y = (size[1] - image_pil.height) // 2
    background.paste(image_pil, (x, y))
    return cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)

def segment_leaf(image):
    image = cv2.resize(image, (512, 512))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 450)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask[:, :, np.newaxis]

def extract_color_histogram(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_features = []
    for i in range(3):
        hist_rgb = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_hsv = cv2.calcHist([image_hsv], [i], None, [256], [0, 256])
        hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_features.extend(hist_rgb)
        hist_features.extend(hist_hsv)
    return np.array(hist_features)

def extract_glcm_features(image):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])

def extract_lbp_texture(image):
    from skimage.feature import local_binary_pattern
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / hist.sum()

def combine_features(image):
    return np.hstack([
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_lbp_texture(image)
    ])

def predict_disease_with_svm(cropped_image, plant_type):
    bundle_dir = os.path.join(BASE_DIR, "Bundle")
    bundle_path = os.path.join(bundle_dir, f"{plant_type}_bundle.pkl")
    if not os.path.exists(bundle_path):
        return f"No SVM model found for {plant_type}"

    bundle = joblib.load(bundle_path)
    svm = bundle['model']
    pca = bundle['pca']
    scaler = bundle['scaler']
    label_map = bundle['label_map']

    segmented = segment_leaf(cropped_image)
    features = combine_features(segmented).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    prediction = svm.predict(features_pca)[0]
    return label_map[prediction]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if original_image is None:
        return jsonify({'error': 'Invalid image'}), 400

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    no_bg = remove(image_rgb, bgcolor=(255, 255, 255, 255), bg_threshold=10)

    if no_bg.shape[2] == 4:
        no_bg = cv2.cvtColor(no_bg, cv2.COLOR_RGBA2RGB)

    no_bg_resized = cv2.resize(no_bg, (640, 640))
    temp_path = os.path.join(BASE_DIR, "temp_image.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(no_bg_resized, cv2.COLOR_RGB2BGR))
    results = yolo_model.predict(temp_path, save=False)
    os.remove(temp_path)

    if results[0].masks is not None and len(results[0].masks.data) > 0:
        masks = results[0].masks.data.cpu().numpy()
        areas = [np.sum(mask) for mask in masks]
        largest_mask = masks[np.argmax(areas)].astype(np.uint8) * 255
        masked_image = cv2.bitwise_and(no_bg_resized, no_bg_resized, mask=largest_mask)
        ys, xs = np.where(largest_mask > 0)
        x1, y1, x2, y2 = max(0, min(xs)-5), max(0, min(ys)-5), min(masked_image.shape[1], max(xs)+5), min(masked_image.shape[0], max(ys)+5)
        cropped = masked_image[y1:y2, x1:x2]
        final_image = resize_with_background(cropped, size=(224, 224))

        img_array = final_image.astype(np.float32)
        img_array = img_array / 127.5 - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = onnx_session.run(None, {input_name: img_array})[0][0]
        top_index = np.argmax(preds)
        plant_type = class_names[top_index]
        diagnosis = predict_disease_with_svm(cropped, plant_type)

        return jsonify({
            'plant_type': plant_type,
            'diagnosis': diagnosis
        })

    return jsonify({'error': 'No mask found'}), 400

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', debug=True)
