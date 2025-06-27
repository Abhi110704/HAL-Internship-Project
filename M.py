import os
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from PIL import Image
import tempfile
from ultralytics import YOLO

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for images
LOGO_PATH = os.path.join(BASE_DIR, "hal_logo.png")

# === PAGE CONFIG ===
st.set_page_config(page_title="🛠️ HAL Defect Detection", layout="centered")

# === GOVERNMENT BANNER ===
st.markdown(
    '''
    <div style="display: flex; justify-content: center;">
        <div style="background-color: #0b3d91; color: white; padding: 0.5em 2em; font-size: 1.3em; font-weight: bold; border-radius: 4px; display: inline-block; text-align: center;">
            Government of India | भारत सरकार
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# === SIDEBAR ===
with st.sidebar:
    st.image(LOGO_PATH, width=190)

st.sidebar.markdown("""
### 🛠️ HAL Defect Detection System

This site provides AI-powered defect detection for aircraft parts using image comparison and deep learning (YOLOv8). Upload your reference and test images to get started.

---
### 🛰️ About HAL
Hindustan Aeronautics Limited (HAL) is an Indian state-owned aerospace and defence company. HAL is involved in the design, fabrication, and assembly of aircraft, jet engines, helicopters, and their spare parts.

---
#### 👩‍💻 Developed by Abhiyanshi Anand and Ishaan Tripathi

---
##### Disclaimer
This is an official government application. Unauthorized access or misuse is prohibited and may be punishable under applicable laws. Data uploaded is used only for defect detection purposes.
""")

# === HEADER ===
# Centered header using HTML in markdown
st.markdown(
    '''
    <div style="text-align: center;">
        <h2 style="margin-bottom: 0.2em;">🛠️ HAL Parts Defect Detection System</h2>
        <h3 style="margin: 0.2em 0;">🛰️ Hindustan Aeronautics Limited (HAL)</h3>
        <h4 style="margin-top: 0.2em;">👩‍💻 Developed by Abhiyanshu Anand and Ishaan Tripathi</h4>
    </div>
    <hr>
    ''',
    unsafe_allow_html=True
)

# === INSTRUCTIONS ===
with st.expander("📖 How to Use", expanded=True):
    st.markdown("""
    1. **Upload a Reference Image** – Ideal version without defects.
    2. **Upload a Test Image** – The image to inspect.
    3. **(Optional)** Upload a YOLOv8 `.pt` model file.
    4. View AI results via image comparison or deep learning detection.
    
    > **Note:** For best results, upload images in the same orientation and with similar zoom (scale). The system will try to auto-align and scale images, but similar field of view improves accuracy and speed.
    """)

# === FUNCTIONS ===

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return thresh

def classify_defect(area, w, h):
    aspect_ratio = w / h
    if area > 2000 and aspect_ratio > 2:
        return "Scratch"
    elif area > 2000:
        return "Dent"
    elif area < 2000:
        return "Crack"
    else:
        return "Unknown"

def detect_defects(ref_img, test_img):
    ref = cv2.resize(ref_img, (512, 512))
    test = cv2.resize(test_img, (512, 512))

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    if np.std(ref_gray) < 30 and np.std(test_gray) < 30:
        ref_proc = preprocess_image(ref)
        test_proc = preprocess_image(test)
    else:
        ref_proc = ref_gray
        test_proc = test_gray

    score, diff = ssim(ref_proc, test_proc, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 220, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = test.copy()
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            boxes.append({
                "x": x, "y": y, "width": w, "height": h,
                "area": area,
                "defect_type": classify_defect(area, w, h)
            })

    total_area = 512 * 512
    defect_area = sum([b['area'] for b in boxes])
    defect_percent = (defect_area / total_area) * 100

    return output_img, boxes, defect_percent

# === ALIGNMENT FUNCTION ===
def align_images(ref_img, test_img):
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_gray, None)

    # Matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = ref_img.shape[:2]
        aligned_test = cv2.warpPerspective(test_img, M, (w, h))
        # Alignment is considered good if there are enough matches
        return aligned_test, len(matches) > 30
    else:
        # Not enough matches, return original
        return test_img, False

# === FILE UPLOADS ===
ref_file = st.file_uploader("📁 Upload Reference Image", type=["jpg", "jpeg", "png"])
test_file = st.file_uploader("🧪 Upload Test Image", type=["jpg", "jpeg", "png"])
model_file = st.file_uploader("🤖 (Optional) Upload YOLOv8 Model (.pt)", type=["pt"])

# === PROCESSING ===
if test_file and (ref_file or model_file):
    ref = cv2.imdecode(np.frombuffer(ref_file.read(), np.uint8), cv2.IMREAD_COLOR) if ref_file else None
    test = cv2.imdecode(np.frombuffer(test_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Align test image to reference if both are available
    if ref is not None and test is not None:
        test_aligned, alignment_good = align_images(ref, test)
    else:
        test_aligned, alignment_good = test, True

    st.subheader("📸 Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        if ref is not None:
            st.image(ref, channels="BGR", caption="🟢 Reference Image")
    with col2:
        st.image(test, channels="BGR", caption="🔍 Test Image")

    if not alignment_good:
        st.warning("⚠️ The uploaded images may have very different zoom/scale or content. Results may not be accurate. Try to upload images with similar field of view and scale.")

    # === YOLO DETECTION ===
    if model_file:
        st.subheader("🧠 YOLOv8 AI Detection")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(model_file.read())
            model_path = tmp.name

        model = YOLO(model_path)
        results = model(test)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        names = model.names

        result_img = test.copy()
        defect_table = []

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls)]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            defect_table.append({
                "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                "confidence": f"{conf:.2f}", "defect_type": label
            })

        st.image(result_img, channels="BGR", caption="📦 YOLOv8 Results")
        if defect_table:
            st.subheader("📋 YOLOv8 Detected Defects")
            df = pd.DataFrame(defect_table)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download YOLO Report (CSV)", csv, "yolo_defect_report.csv", "text/csv")
        else:
            st.success("✅ No defects detected by YOLOv8 🎈")

    # === SSIM COMPARISON ===
    else:
        st.subheader("🧠 AI Detected Defects (Image Comparison)")
        result_img, detected_boxes, defect_percent = detect_defects(ref, test_aligned)
        st.image(result_img, channels="BGR", caption="🔴 Differences Highlighted")

        if defect_percent > 0.5:
            st.error(f"⚠️ Defects Found: {defect_percent:.2f}% of the area")
        else:
            st.success("✅ No major defects detected 🎈")

        if detected_boxes:
            st.subheader("📋 Defect Table")
            df = pd.DataFrame(detected_boxes)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Defect Report (CSV)", csv, "defect_report.csv", "text/csv")

# === FOOTER ===
st.markdown(
    '''
    <hr>
    <div style="text-align: center; color: #888; font-size: 0.95em;">
        &copy; 2024 Hindustan Aeronautics Limited (HAL) | Government of India<br>
        For queries, contact: <a href='mailto:info@hal-india.co.in'>info@hal-india.co.in</a>
    </div>
    ''',
    unsafe_allow_html=True
)
