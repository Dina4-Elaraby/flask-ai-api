import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from ultralytics import YOLO
import joblib
from rembg import remove
from PIL import Image
from flask import Flask, request, jsonify


app = Flask(__name__)

# Load YOLO model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "best.pt"))

# Load ONNX MobileNetV2 plant type classifier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_session = ort.InferenceSession(os.path.join(BASE_DIR, "mobilenetv2_plant_classifier.onnx"))
input_name = onnx_session.get_inputs()[0].name

# Class names for ONNX prediction
class_names = ['apple', 'cherry', 'grapes', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']

# Directory containing input images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "Pepper")

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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bundle_dir = os.path.join(BASE_DIR,"Bundle")
    
    bundle_path = os.path.join(bundle_dir, f"{plant_type}_bundle.pkl")
    if not os.path.exists(bundle_path):
        print(f"❌ No SVM bundle found for plant type: {plant_type}")
        return

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
    predicted_class = label_map[prediction]
    print(f"\U0001fa7a Disease Prediction: {predicted_class}")

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error loading {image_path}")
        continue

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    no_bg = remove(image_rgb, bgcolor=(255, 255, 255, 255), bg_threshold=10)

    if no_bg.shape[2] == 4:
        no_bg = cv2.cvtColor(no_bg, cv2.COLOR_RGBA2RGB)

    no_bg_resized = cv2.resize(no_bg, (640, 640))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    temp_path = os.path.join(BASE_DIR, "temporary", "temp_image.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(no_bg_resized, cv2.COLOR_RGB2BGR))
    results = yolo_model.predict(temp_path, save=False)
    os.remove(temp_path)

    if results[0].masks is not None and len(results[0].masks.data) > 0:
        masks = results[0].masks.data.cpu().numpy()
        areas = [np.sum(mask) for mask in masks]
        largest_mask_idx = np.argmax(areas)
        largest_mask = masks[largest_mask_idx].astype(np.uint8) * 255

        masked_image = cv2.bitwise_and(no_bg_resized, no_bg_resized, mask=largest_mask)
        ys, xs = np.where(largest_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            print(f"Mask found but empty in {image_file}")
            continue

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(masked_image.shape[1], x2 + padding)
        y2 = min(masked_image.shape[0], y2 + padding)

        cropped = masked_image[y1:y2, x1:x2]
        final_image = resize_with_background(cropped, size=(224, 224))

        img_array = final_image.astype(np.float32)
        img_array = img_array / 127.5 - 1.0  # MobileNetV2 preprocessing
        img_array = np.expand_dims(img_array, axis=0)

        preds = onnx_session.run(None, {input_name: img_array})[0][0]
        top3_indices = np.argsort(preds)[-3:][::-1]
        top3_labels = [class_names[i] for i in top3_indices]
        top3_scores = [preds[i] for i in top3_indices]

        print(f"\n\U0001f50d Top 3 plant type predictions for {image_file}:")
        for label, score in zip(top3_labels, top3_scores):
            print(f" - {label}: {score * 100:.2f}%")

        plt.figure(figsize=(4, 4))
        plt.imshow(final_image)
        plt.axis('off')
        title = "\n".join([f"{label}: {score*100:.2f}%" for label, score in zip(top3_labels, top3_scores)])
        plt.title(title)


        ############       Flask API      ###########
        plant_type = top3_labels[0]
        diagnosis= predict_disease_with_svm(cropped, plant_type)
        @app.route('/predict', methods=['POST'])
        def predict():
            
            if 'image' not in request.files:
                return jsonify({'error': 'No image file uploaded'}), 400   
            else:      
               return jsonify({
                'plant_type': plant_type,
                'diagnosis': diagnosis,
               
        })
            
        if __name__ == "__main__":
            port = int(os.environ.get("PORT", 5000))
            app.run(host='0.0.0.0', port=port)
