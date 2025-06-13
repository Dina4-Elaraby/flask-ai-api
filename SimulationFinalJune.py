import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import joblib
from rembg import remove
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)



# Load YOLO model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
# yolo_model = YOLO("C:/wamp64/www/MyGradProject/PlantProject/test_flask_ai_api/best.pt")

# Load MobileNetV2 plant type classifier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classifier_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "final_model_to_identify_plants.keras"))

# classifier_model = tf.keras.models.load_model("C:/wamp64/www/MyGradProject/PlantProject/test_flask_ai_api/final_model_to_identify_plants.keras")
# Class names for MobileNetV2 prediction
class_names = ['apple', 'cherry', 'grapes', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']

# Directory containing input images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "Pepper")

#input_dir = "C:/wamp64/www/MyGradProject/PlantProject/test_flask_ai_api/Pepper"

# Function to resize and center image on white background
def resize_with_background(image, size=(224, 224), background_color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_pil.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', size, background_color)
    x = (size[0] - image_pil.width) // 2
    y = (size[1] - image_pil.height) // 2
    background.paste(image_pil, (x, y))
    return cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)

# Function to segment leaf
def segment_leaf(image):
    image = cv2.resize(image, (512, 512))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 450)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask[:, :, np.newaxis]

# Extract Color Histogram
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

# Extract GLCM Features
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

# Extract LBP Texture
def extract_lbp_texture(image):
    from skimage.feature import local_binary_pattern
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / hist.sum()

# Combine all features
def combine_features(image):
    return np.hstack([
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_lbp_texture(image)
    ])

# Predict disease using SVM bundle
def predict_disease_with_svm(cropped_image, plant_type):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bundle_dir = os.path.join(BASE_DIR, "Bundle")
    
    # bundle_dir = "C:/wamp64/www/MyGradProject/PlantProject/test_flask_ai_api/Bundle"
    bundle_path = os.path.join(bundle_dir, f"{plant_type}_bundle.pkl")
    if not os.path.exists(bundle_path):
        print(f"❌ No SVM bundle found for plant type: {plant_type}")
        return

    # Load SVM, PCA, Scaler, and Label Map
    bundle = joblib.load(bundle_path)
    svm = bundle['model']
    pca = bundle['pca']
    scaler = bundle['scaler']
    label_map = bundle['label_map']

    # Preprocess image
    segmented = segment_leaf(cropped_image)
    features = combine_features(segmented).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Predict
    prediction = svm.predict(features_pca)[0]
    predicted_class = label_map[prediction]
    print(f"🩺 Disease Prediction: {predicted_class}")
    #edit
    return predicted_class

# ---------------------------- Main Processing Loop ----------------------------

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error loading {image_path}")
        continue

    # Remove background
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    no_bg = remove(image_rgb, bgcolor=(255, 255, 255, 255), bg_threshold=10)

    if no_bg.shape[2] == 4:
        no_bg = cv2.cvtColor(no_bg, cv2.COLOR_RGBA2RGB)

    # Resize to 640x640 for YOLO
    no_bg_resized = cv2.resize(no_bg, (640, 640))

    # Save temporarily and run YOLO
    Base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_path = os.path.join(Base_dir, "temporary", "temp_image.jpg")
    
    # temp_path = "C:/wamp64/www/EnvironmentGradProject/PlantProjectEnvirnment/test_flask_ai_api/temporary/temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(no_bg_resized, cv2.COLOR_RGB2BGR))
    results = yolo_model.predict(temp_path, save=False)
    os.remove(temp_path)

    # Ensure masks exist
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        masks = results[0].masks.data.cpu().numpy()  # (num_instances, H, W)
        areas = [np.sum(mask) for mask in masks]
        largest_mask_idx = np.argmax(areas)
        largest_mask = masks[largest_mask_idx].astype(np.uint8) * 255

        # Apply mask to the image
        masked_image = cv2.bitwise_and(no_bg_resized, no_bg_resized, mask=largest_mask)

        # Extract bounding box from the mask
        ys, xs = np.where(largest_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            print(f"Mask found but empty in {image_file}")
            continue

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Add padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(masked_image.shape[1], x2 + padding)
        y2 = min(masked_image.shape[0], y2 + padding)

        # Crop the image using the mask
        cropped = masked_image[y1:y2, x1:x2]

        # Resize and center on white background
        final_image = resize_with_background(cropped, size=(224, 224))

        # Prepare image for classification
        img_array = image.img_to_array(final_image)
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

        # Predict class
        preds = classifier_model.predict(img_batch, verbose=0)[0]
        top3_indices = np.argsort(preds)[-3:][::-1]  # Top 3 indices
        top3_labels = [class_names[i] for i in top3_indices]
        top3_scores = [preds[i] for i in top3_indices]

        # Print top 3 predictions
        print(f"\n🔍 Top 3 plant type predictions for {image_file}:")
        for label, score in zip(top3_labels, top3_scores):
            print(f" - {label}: {score * 100:.2f}%")

        # Display image with top prediction
        plt.figure(figsize=(4, 4))
        plt.imshow(final_image)
        plt.axis('off')
        title = "\n".join([f"{label}: {score*100:.2f}%" for label, score in zip(top3_labels, top3_scores)])


        ###########       Flask API      ###########
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
       
        if __name__ == '__main__':
           app.run(debug=True)
       


    
      