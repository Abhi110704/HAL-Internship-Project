import os
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from PIL import Image
import tempfile
from ultralytics import YOLO
import sys
import skimage.exposure
from skimage import color as skcolor
from skimage.restoration import denoise_bilateral
from scipy.ndimage import binary_opening, binary_closing
from skimage import measure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "hal_logo.png")
st.set_page_config(page_title="🛠️ HAL Parts Defect Detection 🛠️", layout="centered")
with st.sidebar:
    st.image(LOGO_PATH, width=190)
    st.markdown("#### Detection Sensitivity Controls")
    ssim_thresh = st.slider("SSIM Defect Threshold (lower = more sensitive)", min_value=0, max_value=255, value=220)
    color_thresh = st.slider("Color Defect Threshold (Delta E)", min_value=1, max_value=50, value=15)
    pattern_min_matches = st.slider("Pattern Min Matches", min_value=5, max_value=50, value=10)
    yolo_conf_thresh = st.slider("YOLO Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    st.markdown("---")
    st.markdown("""
### 🛰️ About HAL
Hindustan Aeronautics Limited (HAL) is an Indian state-owned aerospace and defence company. HAL is involved in the design, fabrication, and assembly of aircraft, jet engines, helicopters, and their spare parts.

---
#### 👩‍💻 Developed by Abhiyanshu Anand and Ishaan Tripathi 👩‍💻
""")
    st.markdown("---")
st.markdown(
    '''
    <div style="text-align: center;">
        <h2 style="margin-bottom: 0.2em;">🛠️ HAL Parts Defect Detection System 🛠️</h2>
        <h3 style="margin: 0.2em 0;">🛰️ Hindustan Aeronautics Limited (HAL)</h3>
        <h4 style="margin-top: 0.2em;">👩‍💻 Developed by Abhiyanshu Anand and Ishaan Tripathi 👩‍💻</h4>
    </div>
    <hr>
    ''',
    unsafe_allow_html=True
)
color_toggle = st.checkbox("Enable Color Defect Detection", value=True)
deltae_toggle = st.checkbox("Enable DeltaE (LAB) Color Defect Detection", value=True)
pattern_toggle = st.checkbox("Enable Pattern Defect Detection", value=True)

# Robust preprocessing for any image

def robust_preprocess(img):
    if img is None:
        return None
    # Convert grayscale to BGR
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

def auto_rotate_image(test_img, ref_img):
    best_img = test_img
    best_score = -1
    ref_h, ref_w = ref_img.shape[:2]
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = test_img
        else:
            rotated = cv2.rotate(test_img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        rotated_resized = cv2.resize(rotated, (ref_w, ref_h))
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(rotated_resized, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(ref_gray, test_gray, full=True)
        if score > best_score:
            best_score = score
            best_img = rotated_resized
    return best_img

def align_images(ref_img, test_img):
    # Try ORB feature-based alignment
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            h, w = ref_img.shape[:2]
            aligned_test = cv2.warpPerspective(test_img, M, (w, h))
            return aligned_test
    # Fallback: auto-rotate
    return auto_rotate_image(test_img, ref_img)

def draw_defect_boundaries(image, mask, color=(0, 0, 255), thickness=5):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = (image * 0.6).astype(np.uint8)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 3)
    return overlay

def detect_ssim_defects(ref_img, test_img, ssim_thresh=220):
    ref = cv2.resize(ref_img, (512, 512))
    test = cv2.resize(test_img, (512, 512))
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(ref_gray, test_gray, full=True)
    diff = (diff * 255).astype("uint8")
    kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    mask = cv2.threshold(diff, ssim_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    mask = binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((5,5))).astype(np.uint8)
    area = np.sum(mask > 0)
    percent = (area / (512*512)) * 100
    marked = draw_defect_boundaries(test, mask, color=(0,0,255), thickness=5)
    return marked, mask, percent

def detect_color_defects_with_map(ref_img, test_img, color_thresh=15):
    ref_lab = skcolor.rgb2lab(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    test_lab = skcolor.rgb2lab(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    delta_e = skcolor.deltaE_ciede2000(ref_lab, test_lab)
    mask = (delta_e > color_thresh).astype(np.uint8)
    mask = binary_opening(mask, structure=np.ones((5,5))).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((7,7))).astype(np.uint8)
    area = np.sum(mask > 0)
    percent = (area / mask.size) * 100
    marked = draw_defect_boundaries(test_img, mask, color=(255,0,255), thickness=5)
    # Normalize deltaE map for visualization
    delta_e_norm = ((delta_e - np.min(delta_e)) / (np.ptp(delta_e) + 1e-8) * 255).astype(np.uint8)
    delta_e_color = cv2.applyColorMap(delta_e_norm, cv2.COLORMAP_JET)
    # Resize deltaE map to match marked image size
    if delta_e_color.shape[:2] != marked.shape[:2]:
        delta_e_color = cv2.resize(delta_e_color, (marked.shape[1], marked.shape[0]), interpolation=cv2.INTER_NEAREST)
    return marked, mask*255, percent, delta_e_color

def detect_pattern_defects(ref_img, test_img, min_matches=10):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    if des1 is None or des2 is None:
        return test_img, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pattern_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, matches[:min_matches], None, flags=2)
    num_matches = len(matches)
    return pattern_img, num_matches

ref_files = st.file_uploader("📁 Upload Reference Images (Multi-Angle)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
test_files = st.file_uploader("🧪 Upload Test Images (Multi-Angle)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
model_file = st.file_uploader("🤖 (Optional) Upload YOLOv8 Model (.pt)", type=["pt"])

if ref_files and test_files:
    any_defect = False
    summary_rows = []
    test_file_bytes_list = []
    test_file_names = []
    for test_file in test_files:
        test_file.seek(0)
        test_bytes = test_file.read()
        test_file_bytes_list.append(test_bytes)
        test_file_names.append(test_file.name)
    for ref_idx, ref_file in enumerate(ref_files):
        ref_bytes = ref_file.read()
        if not ref_bytes:
            st.error(f"Reference image file {ref_file.name} is empty or could not be read.")
            continue
        ref = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        if ref is None:
            st.error(f"Uploaded reference image {ref_file.name} is not a valid image file.")
            continue
        for test_idx, test_bytes in enumerate(test_file_bytes_list):
            if not test_bytes:
                st.error(f"Test image file {test_file_names[test_idx]} is empty or could not be read.")
                continue
            test = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)
            if test is None:
                st.error(f"Uploaded test image {test_file_names[test_idx]} is not a valid image file.")
                continue
            st.markdown(f"## 🖼️ Reference {ref_idx+1} vs Test {test_idx+1}")
            if ref is not None and test is not None:
                test = auto_rotate_image(test, ref)
            test_aligned = align_images(ref, test)
            st.subheader("📸 Uploaded Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(ref, channels="BGR", caption=f"🟢 Reference Image {ref_idx+1}")
            with col2:
                st.image(test, channels="BGR", caption=f"🔍 Test Image {test_idx+1}")
            st.info(f"Alignment method used: ORB feature-based alignment")
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
                    if conf < yolo_conf_thresh:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    label = names[int(cls)]
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(result_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    defect_table.append({
                        "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                        "confidence": f"{conf:.2f}", "defect_type": label
                    })
                st.subheader("📸 Side-by-Side Comparison (YOLO)")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref, channels="BGR", caption=f"🟢 Reference Image {ref_idx+1}")
                with col2:
                    st.image(result_img, channels="BGR", caption="📦 YOLOv8 Results")
                if defect_table:
                    st.subheader("📋 YOLOv8 Detected Defects")
                    df = pd.DataFrame(defect_table)
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode()
                    st.download_button(f"⬇️ Download YOLO Report (CSV) - Ref{ref_idx+1}_Test{test_idx+1}", csv, f"yolo_defect_report_ref{ref_idx+1}_test{test_idx+1}.csv", "text/csv")
                    any_defect = True
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "YOLO", "Defect": True})
                else:
                    st.success("✅ No defects detected by YOLOv8 🎈")
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "YOLO", "Defect": False})
            else:
                st.subheader("🧠 AI Detected Defects (Image Comparison)")
                ssim_marked, ssim_mask, ssim_percent = detect_ssim_defects(ref, test_aligned, ssim_thresh=ssim_thresh)
                st.subheader("SSIM Defect Detection")
                # Set a slightly smaller display size for all images
                image_width = 210
                image_height = 210
                # Resize all SSIM images to the same size
                ssim_marked_resized = cv2.resize(ssim_marked, (image_width, image_height), interpolation=cv2.INTER_AREA)
                ssim_mask_resized = cv2.resize(ssim_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                ssim_diff = cv2.absdiff(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(test_aligned, cv2.COLOR_BGR2GRAY))
                ssim_diff_resized = cv2.resize(ssim_diff, (image_width, image_height), interpolation=cv2.INTER_AREA)
                # Create a red overlay for the SSIM defect mask
                ssim_mask_color = np.zeros_like(ssim_marked_resized)
                if len(ssim_mask_resized.shape) == 2:
                    mask_bool = ssim_mask_resized > 0
                    ssim_mask_color[mask_bool] = [255, 0, 0]  # Red for defects
                # Blend with the test image for semi-transparent overlay
                ssim_overlay = cv2.addWeighted(ssim_marked_resized, 0.7, ssim_mask_color, 0.6, 0)
                # SSIM display with overlay
                st.subheader("SSIM Defect Detection")
                ssim_col1, ssim_spacer1, ssim_col2, ssim_spacer2, ssim_col3 = st.columns([1,0.1,1,0.1,1])
                with ssim_col1:
                    st.image(ssim_marked_resized, channels="BGR", caption=f"SSIM Marked (Area: {ssim_percent:.2f}%)", width=image_width)
                with ssim_col2:
                    st.image(ssim_overlay, channels="BGR", caption="SSIM Defect Mask (Overlay)", width=image_width)
                with ssim_col3:
                    st.image(ssim_diff_resized, caption="SSIM Diff Image", clamp=True, width=image_width)
                # Color/DeltaE
                color_marked, color_mask, color_percent, delta_e_map = detect_color_defects_with_map(ref, test_aligned, color_thresh=color_thresh)
                color_marked_resized = cv2.resize(color_marked, (image_width, image_height), interpolation=cv2.INTER_AREA)
                color_mask_resized = cv2.resize(color_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                delta_e_map_resized = cv2.resize(delta_e_map, (image_width, image_height), interpolation=cv2.INTER_AREA)
                st.subheader("Color/DeltaE Defect Detection")
                color_col1, color_spacer1, color_col2, color_spacer2, color_col3 = st.columns([1,0.1,1,0.1,1])
                with color_col1:
                    st.image(color_marked_resized, channels="BGR", caption=f"Color/DeltaE Marked (Area: {color_percent:.2f}%)", width=image_width)
                with color_col2:
                    st.image(color_mask_resized, caption="Color/DeltaE Defect Mask", clamp=True, width=image_width)
                with color_col3:
                    st.image(delta_e_map_resized, caption="DeltaE Map", clamp=True, width=image_width)
                if ssim_percent > 0.5:
                    st.error(f"⚠️ Defects Found: {ssim_percent:.2f}% of the area")
                    any_defect = True
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "SSIM", "Defect": True})
                else:
                    st.success("✅ No major defects detected 🎈")
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "SSIM", "Defect": False})
                if color_percent > 0.5:
                    st.error(f"⚠️ Color Defects Found: {color_percent:.2f}% of the area")
                    any_defect = True
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Color", "Defect": True})
                else:
                    st.success("✅ No major color defects detected 🎈")
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Color", "Defect": False})
                if pattern_toggle:
                    st.subheader("🔳 Pattern Defect Detection (ORB)")
                    pattern_img, num_matches = detect_pattern_defects(ref, test_aligned, min_matches=pattern_min_matches)
                    st.image(pattern_img, channels="BGR", caption=f"Pattern Matching (ORB) - Matches: {num_matches}")
                    if num_matches < pattern_min_matches:
                        st.error("⚠️ Pattern mismatch detected!")
                        any_defect = True
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Pattern", "Defect": True})
                    else:
                        st.success("✅ Pattern matches are sufficient 🎈")
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Pattern", "Defect": False})
    st.markdown("---")
    if summary_rows:
        st.subheader("📝 Multi-Angle, Multi-Reference Summary Table")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df)
        csv = summary_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download All Results (CSV)", csv, "multi_angle_multi_reference_summary.csv", "text/csv")
    if any_defect:
        st.error("❗ Defect(s) detected in one or more reference/test image pairs.")
    else:
        st.success("✅ No defects detected in any reference/test image pair!")
