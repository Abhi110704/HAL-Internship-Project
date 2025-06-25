import streamlit as st
import pandas as pd
import os
import numpy as np
import cv2
import base64
from PIL import Image
from scipy.spatial.distance import cosine
from io import BytesIO

# Optional YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# -------------------- Config --------------------
st.set_page_config(page_title="HAL Aircraft Part Side Detector", layout="wide")

CORRECTION_FILE = "corrections.csv"
TEST_IMAGE_PATH = "5994af38-34de-44bc-93a2-a40e136cad3b.png"
LOGO_PATH = "hal_logo.png"

# -------------------- Helper Functions --------------------

@st.cache_data
def load_corrections():
    if os.path.exists(CORRECTION_FILE):
        return pd.read_csv(CORRECTION_FILE)
    return pd.DataFrame(columns=["Image", "Corrected Side"])

def save_correction(image_name, corrected_side):
    df = load_corrections()
    df = df[df.Image != image_name]
    df.loc[len(df.index)] = [image_name, corrected_side]
    df.to_csv(CORRECTION_FILE, index=False)

def correct_image_rotation(image):
    arr = np.array(image.convert("L"))
    arr = cv2.GaussianBlur(arr, (5, 5), 0)
    edges = cv2.Canny(arr, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            deg = np.rad2deg(theta)
            if 80 < deg < 100 or 260 < deg < 280:
                continue
            angle = theta
            break
    angle = np.rad2deg(angle) - 90
    return image.rotate(-angle, expand=True) if abs(angle) > 1 else image

def predict_side_by_symmetry(gray_crop):
    h, w = gray_crop.shape
    w_half = min(w // 2, w - w // 2)
    left_half = gray_crop[:, :w_half]
    right_half = np.fliplr(gray_crop[:, -w_half:])
    left_profile = left_half.mean(axis=0)
    right_profile = right_half.mean(axis=0)
    similarity = 1 - cosine(left_profile, right_profile)
    return ("🅻 Left" if similarity < 0.9 else "🆁 Right", similarity)

def advanced_confidence_score(gray_crop):
    total = np.sum(gray_crop)
    indices = np.arange(gray_crop.shape[1])[None, :]
    com = np.sum(gray_crop * indices) / total if total != 0 else gray_crop.shape[1] // 2
    dist_to_center = abs(com - (gray_crop.shape[1] / 2))
    norm_dist = dist_to_center / (gray_crop.shape[1] / 2)
    return round(max(0, 1 - norm_dist), 3)

def simulate_detection(img, name, corrections, prefer_symmetry, model=None, is_2d_diagram=False):
    arr = np.array(img)
    h, w = arr.shape[:2]
    img_center_x = w // 2

    # Try YOLO detection
    if model:
        results = model(arr)
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0], 'boxes') else []
        x1, y1, x2, y2 = map(int, boxes[0]) if len(boxes) > 0 else (50, 50, w - 50, h - 50)
    else:
        x1, y1, x2, y2 = 50, 50, w - 50, h - 50

    crop = img.crop((x1, y1, x2, y2)).convert("L")
    gray = np.array(crop)

    # 2D Diagram robust logic
    if is_2d_diagram:
        # Use bounding box center for orientation
        cx = (x1 + x2) // 2
        margin = int(w * 0.05)
        if cx < (w // 2 - margin):
            predicted = "🅻 Left"
        elif cx > (w // 2 + margin):
            predicted = "🆁 Right"
        else:
            predicted = "Center"
        com_side = predicted
        symmetry_side = predicted
        sim_score = 1.0
        confidence = 1.0
        com_x = cx
    else:
        total = np.sum(gray)
        com_x = (x1 + x2) / 2 if total == 0 else np.sum(gray * np.arange(gray.shape[1])[None, :]) / total + x1
        com_side = "🅻 Left" if com_x < img_center_x else "🆁 Right"
        symmetry_side, sim_score = predict_side_by_symmetry(gray)
        confidence = advanced_confidence_score(gray)
        predicted = symmetry_side if prefer_symmetry else com_side

    corrected = corrections[corrections.Image == name]
    final = corrected["Corrected Side"].values[0] if not corrected.empty else predicted

    return {
        "Image": name,
        "Predicted Side": predicted,
        "Final Side": final,
        "Confidence Score": confidence,
        "X-Coordinate": int(com_x),
        "Symmetry Score": round(sim_score, 3) if not is_2d_diagram else 1.0,
        "COM Side": com_side,
        "Symmetry Side": symmetry_side
    }

def display_logo():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{logo_b64}' width='160'>
        </div>
        <h1 style='text-align: center;'>✈️ HAL Aircraft Part Side Identifier</h1>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 'hal_logo.png' not found. Please place it in the same folder.")

# -------------------- UI --------------------

display_logo()

st.markdown("""
### 📘 How to Use
1. Upload your YOLOv8 `.pt` model (optional).
2. Upload part images or take a picture.
3. Enable "Auto-Rotate" for tilted parts.
4. COM and Symmetry-based prediction is done.
5. Manually correct if needed.
6. Download your results.
""")

st.sidebar.header("⚙️ Input Settings")
input_mode = st.sidebar.radio("Select Image Source", ["Upload Images", "Use Camera"])
rotate_option = st.sidebar.checkbox("🔄 Auto-Rotate Images")
prefer_symmetry = st.sidebar.checkbox("🧠 Prefer Symmetry Over COM")
only_final = st.sidebar.checkbox("📌 Show Only Final Prediction")
# Add 2D Diagram Mode
is_2d_diagram = st.sidebar.checkbox("2D Diagram Mode (for sketches/blueprints)")

st.sidebar.header("📦 Model Upload")
model_file = st.sidebar.file_uploader("Upload YOLOv8 .pt Model (optional)", type="pt")
filter_side = st.selectbox("Filter by side", ["All", "Left", "Right"])
zoom = st.sidebar.slider("Zoom (Preview Scale %)", 25, 200, 100)

model = None
if model_file and YOLO_AVAILABLE:
    with st.spinner("Loading YOLOv8 model..."):
        try:
            model = YOLO(model_file)
            st.success("✅ Model loaded successfully.")
        except Exception as e:
            st.warning(f"⚠️ Failed to load model: {e}")

# -------------------- Image Handling --------------------

images, image_names = [], []

if os.path.exists(TEST_IMAGE_PATH):
    img = Image.open(TEST_IMAGE_PATH)
    img.thumbnail((600, 600))
    if rotate_option:
        img = correct_image_rotation(img)
    images.append(img)
    image_names.append("test_hardware.png")

if input_mode == "Upload Images":
    uploaded_files = st.sidebar.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    for file in uploaded_files or []:
        img = Image.open(file)
        img.thumbnail((600, 600))
        if rotate_option:
            img = correct_image_rotation(img)
        images.append(img)
        image_names.append(file.name)

elif input_mode == "Use Camera":
    cam_img = st.camera_input("Take a Picture")
    if cam_img:
        img = Image.open(cam_img)
        img.thumbnail((600, 600))
        if rotate_option:
            img = correct_image_rotation(img)
        images.append(img)
        image_names.append("camera_capture.jpg")

# -------------------- Predictions --------------------

corrections = load_corrections()
results = []

if images:
    with st.expander("📸 Image Preview Grid", expanded=True):
        for i in range(0, len(images), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(images):
                    img = images[i + j]
                    scaled = img.copy()
                    w, h = scaled.size
                    scaled = scaled.resize((int(w * zoom / 100), int(h * zoom / 100)))
                    pred = simulate_detection(scaled, image_names[i + j], corrections, prefer_symmetry, model, is_2d_diagram)
                    with cols[j]:
                        st.image(scaled, caption=f"{image_names[i + j]}\nPrediction: {pred['Final Side']}", use_container_width=True)

    for img, name in zip(images, image_names):
        result = simulate_detection(img, name, corrections, prefer_symmetry, model, is_2d_diagram)
        results.append(result)
        with st.form(f"correction_{name}"):
            st.write(f"✍️ Suggest correction for: **{name}**")
            corrected = st.selectbox("Choose side:", ["", "Left", "Right"], key=name)
            submit = st.form_submit_button("Save Correction")
            if submit and corrected:
                save_correction(name, corrected)
                st.success(f"✅ Correction saved for {name} as {corrected}")
                st.rerun()

if results:
    df = pd.DataFrame(results)
    # Add Manual Correction and Final Prediction columns
    corrections_map = {row['Image']: row['Corrected Side'] for _, row in load_corrections().iterrows()}
    df['Manual Correction'] = df['Image'].map(corrections_map).fillna("")
    df['Final Prediction'] = df['Manual Correction'].where(df['Manual Correction'] != "", df['Predicted Side'])

    # Reorder columns for clarity
    display_cols = [
        'Image',
        'Predicted Side',
        'Manual Correction',
        'Final Prediction',
        'Confidence Score',
        'X-Coordinate',
        'Symmetry Score',
        'COM Side',
        'Symmetry Side'
    ]
    df = df[display_cols]

    if filter_side != "All":
        df = df[df['Final Prediction'].str.contains(filter_side)]
    if only_final:
        df = df[['Image', 'Final Prediction', 'Confidence Score']]

    st.markdown("### ✅ Final Prediction Summary")
    st.dataframe(df, use_container_width=True)

    corrected_df = df[df['Manual Correction'] != ""]
    st.download_button("📥 Download CSV", corrected_df.to_csv(index=False).encode(), "HAL_Part_Detection.csv")
else:
    st.info("📸 Please upload or capture an image to begin analysis.")

# -------------------- Footer --------------------
st.markdown("""
---
<div style='text-align: center;'>
    🛠️ Developed with by <strong>Abhiyanshu Anand</strong>
</div>
""", unsafe_allow_html=True)
