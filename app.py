# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import os
# from skimage.feature import graycomatrix, graycoprops
# import matplotlib.pyplot as plt

# # ========== MODEL LOADING ==========
# leaf_type_model = load_model("leaf_type.h5", compile=False)
# model_dict = {
#     "apple": load_model("E:/DL_MODELS/model_leaf_apple.h5"),
#     "mango": load_model("E:/DL_MODELS/model_leaf_mango.h5"),
#     "cotton": load_model("E:/DL_MODELS/model_leaf_cotton.h5"),
#     "potato": load_model("E:/DL_MODELS/model_leaf_potato.h5"),
#     "grape": load_model("E:/DL_MODELS/model_leaf_grape.h5"),
#     "tomato": load_model("E:/DL_MODELS/model_leaf_tomato.h5"),
# }

# # ========== BACKGROUND REMOVAL FUNCTION ==========
# def remove_background(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     result = cv2.bitwise_and(img, img, mask=mask)
#     return result, mask

# # ========== SEVERITY ESTIMATION ==========
# def calculate_severity(mask):
#     total_pixels = mask.size
#     diseased_pixels = cv2.countNonZero(mask)
#     percentage = (diseased_pixels / total_pixels) * 100
#     return round(percentage, 2)

# # ========== GLCM FEATURE EXTRACTION ==========
# def extract_glcm_features(gray_img):
#     gray_img = np.uint8(gray_img / 4)
#     glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=64, symmetric=True, normed=True)
#     features = {
#         'Contrast': round(graycoprops(glcm, 'contrast')[0, 0], 4),
#         'Dissimilarity': round(graycoprops(glcm, 'dissimilarity')[0, 0], 4),
#         'Homogeneity': round(graycoprops(glcm, 'homogeneity')[0, 0], 4),
#         'Energy': round(graycoprops(glcm, 'energy')[0, 0], 4),
#         'Correlation': round(graycoprops(glcm, 'correlation')[0, 0], 4),
#         'ASM': round(graycoprops(glcm, 'ASM')[0, 0], 4)
#     }
#     return features

# # ========== LEAF TYPE PREDICTION ==========
# def predict_leaf_type(image_array):
#     img = cv2.resize(image_array, (224, 224)) / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = leaf_type_model.predict(img)
#     classes = ['apple', 'cotton', 'grape', 'mango', 'potato', 'tomato']
#     return classes[np.argmax(pred)]

# # ========== DISEASE PREDICTION ==========
# def predict_disease(image_array, model):
#     img = cv2.resize(image_array, (224, 224)) / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     return pred

# # ========== CREATE RED HIGHLIGHT OVERLAY ==========
# def highlight_diseased_area(image, mask):
#     mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
#     mask = mask.astype(np.uint8)

#     # Create red overlay where disease is detected
#     red_overlay = np.zeros_like(image)
#     red_overlay[:, :] = [255, 0, 0]  # Red color

#     # Apply mask to red overlay
#     red_disease = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)

#     # Combine original image and red-diseased part
#     highlighted = cv2.addWeighted(image, 1, red_disease, 0.5, 0)
#     return highlighted

# # ========== STREAMLIT UI ==========
# st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
# st.title("🌿 Leaf Disease Detection & Severity Estimation")

# uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("📷 Original Image")
#         st.image(image_np, use_column_width=True)

#     # Remove background
#     result_img, mask = remove_background(image_np)

#     with col2:
#         st.subheader("🧼 Background Removed")
#         st.image(result_img, use_column_width=True)

#     # Predict leaf type
#     leaf_type = predict_leaf_type(result_img)
#     st.success(f"🌿 Detected Leaf Type: **{leaf_type.upper()}**")

#     # Predict disease
#     disease_model = model_dict.get(leaf_type)
#     if disease_model:
#         disease_pred = predict_disease(result_img, disease_model)
#         class_names = ['Healthy', 'Diseased']
#         predicted_class = class_names[np.argmax(disease_pred)]
#         confidence = np.max(disease_pred) * 100
#         st.info(f"🦠 Disease Prediction: **{predicted_class}** ({confidence:.2f}%)")

#         # Severity
#         severity = calculate_severity(mask)
#         st.warning(f"🔥 Severity Estimated: **{severity:.2f}%** of the leaf is affected.")

#         # GLCM features
#         gray_leaf = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
#         glcm_features = extract_glcm_features(gray_leaf)

#         st.subheader("📊 GLCM Texture Features")
#         for k, v in glcm_features.items():
#             st.write(f"- **{k}**: {v}")
        
#         # Processed image with red disease highlight
#         highlighted_img = highlight_diseased_area(image_np, mask)
#         st.subheader("📌 Processed Image with Disease Highlight")
#         st.image(highlighted_img, use_column_width=True)

#         # Before/After comparison
#         st.subheader("🆚 Before vs After")
#         comp_col1, comp_col2 = st.columns(2)
#         with comp_col1:
#             st.image(image_np, caption="Original", use_column_width=True)
#         with comp_col2:
#             st.image(highlighted_img, caption="Disease Highlighted", use_column_width=True)
#     else:
#         st.error("No disease model available for the detected leaf type.")
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
import joblib
import warnings
import matplotlib.pyplot as plt
import rembg
import os
# ========== Class Mapping ==========
disease_class_indices_map = {
    "apple": {0: "Alternia leaf spot", 1: "Brown Spot", 2: "Fray spot", 3: "Healthy", 4: "Rust"},
    "cotton": {0: "bacterial blight", 1: "curl virus", 2: "fussarium wilt", 3: "healthy"},
    "grape": {0: "Black rot", 1: "Esca", 2: "Healthy", 3: "Leaf Blight"},
    "mango": {
        0: "Anthracnose", 1: "Bacterial Canker", 2: "Cutting Weevil", 3: "Dieback",
        4: "Gall Midge", 5: "Healthy", 6: "Powdery Mildew", 7: "Sooty Mould"
    },
    "potato": {0: "Bacteria", 1: "Fungi", 2: "Healthy", 3: "Nematode", 4: "Pest", 5: "Phytophthora", 6: "Virus"},
    "tomato": {
        0: "Bacterial spot", 1: "Early blight", 2: "Late blight", 3: "Leaf Mold",
        4: "Septoria", 5: "Spider mites", 6: "Target Spot", 7: "TYLCV",
        8: "TMV", 9: "Healthy"
    }
}

color_ranges = {
    "green": ((36, 25, 25), (86, 255, 255)),
    "yellow": ((22, 93, 0), (45, 255, 255)),
    "brown": ((10, 100, 20), (20, 255, 200)),
    "orange": ((10, 100, 20), (25, 255, 255)),
    "skin": ((0, 48, 80), (20, 255, 255)),
    "gray": ((0, 0, 50), (180, 50, 200)),
    "red": ((0, 100, 100), (10, 255, 255))
}

# ========== Utility Functions ==========
@st.cache_resource
def load_leaf_type_model():
    return load_model("leaf_type.h5", compile=False)

@st.cache_resource
def load_disease_model(leaf_type):
    model_path = os.path.join("E:/DL_MODELS", f"model_leaf_{leaf_type}.h5")
    return load_model(model_path)

@st.cache_resource
def load_glcm_regressor():
    return joblib.load(r"C:\Users\Arunava Chakraborty\Desktop\Plant-Disease-Prediction\glcm_regressor.pkl")

def preprocess_image(img):
    # Convert PIL Image to NumPy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Resize, normalize, and expand dimensions
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    return img


def predict_image_class(model, image, class_indices):
    processed = preprocess_image(image)
    pred = model.predict(processed)
    return class_indices[np.argmax(pred)]

def remove_background(image):
    input_array = np.array(image)
    output_array = rembg.remove(input_array)
    return Image.fromarray(output_array)

def analyze_leaf(image, color_ranges):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, *color_ranges['green'])

    masks = [cv2.inRange(hsv, color_ranges[k][0], color_ranges[k][1])
             for k in ['yellow', 'brown', 'orange', 'skin', 'gray', 'red']]

    mask_combined = masks[0]
    for m in masks[1:]:
        mask_combined = cv2.bitwise_or(mask_combined, m)

    kernel = np.ones((3, 3), np.uint8)
    mask_combined = cv2.dilate(mask_combined, kernel, iterations=1)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_combined))

    green_pixels = np.sum(mask_green == 255)
    disease_pixels = np.sum(mask_combined == 255)
    disease_percentage = (disease_pixels / (green_pixels + disease_pixels + 1e-5)) * 100

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    diseased_region = cv2.bitwise_and(gray_image, gray_image, mask=mask_combined)
    glcm = graycomatrix(diseased_region, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'asm': graycoprops(glcm, 'ASM')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
    }

    return mask_green, mask_combined, disease_percentage, features

# ========== Streamlit UI ==========
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Leaf Disease Detection and Severity Estimation")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("🧼 Removing background..."):
        bg_removed = remove_background(image)

    st.image(bg_removed, caption="Background Removed", use_column_width=True)

    with st.spinner("🔍 Detecting leaf type..."):
        leaf_type_model = load_leaf_type_model()
        leaf_type = predict_image_class(leaf_type_model, bg_removed, {i: k for i, k in enumerate(disease_class_indices_map)})

    st.success(f"🌿 Leaf Type: **{leaf_type.upper()}**")

    with st.spinner("🦠 Predicting disease..."):
        disease_model = load_disease_model(leaf_type)
        disease = predict_image_class(disease_model, bg_removed, disease_class_indices_map[leaf_type])
    st.info(f"🧬 Detected Disease: **{disease}**")

    with st.spinner("📈 Estimating severity..."):
        mask_green, mask_combined, disease_percentage, features = analyze_leaf(bg_removed, color_ranges)

        glcm_model = load_glcm_regressor()
        input_features = np.array([[features[f] for f in ['contrast', 'correlation', 'energy', 'homogeneity', 'asm', 'dissimilarity']]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcm_pred = glcm_model.predict(input_features)[0]

        severity_score = (glcm_pred + disease_percentage) / 2
        severity_score = min(100, max(0, severity_score))

    st.warning(f"🔥 Severity Estimate: **{severity_score:.2f}%**")

    st.subheader("📊 GLCM Texture Features")
    for k, v in features.items():
        st.write(f"**{k.capitalize()}**: {v:.4f}")

    st.subheader("🖼️ Visual Comparison")
    disease_overlay = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
    disease_overlay[np.where((disease_overlay == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    overlay_image = cv2.addWeighted(cv2.cvtColor(np.array(bg_removed), cv2.COLOR_RGB2BGR), 1.0, disease_overlay, 0.5, 0)
    st.image(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), caption="Disease Overlay", use_column_width=True)

# End of app.py
