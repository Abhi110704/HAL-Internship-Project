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
    st.markdown("""
### 🛠️ HAL Defect Detection System

This site provides AI-powered defect detection for aircraft parts using image comparison and deep learning (YOLOv8). Upload your reference and test images to get started.

---
""")
    # === LIMIT CONTROLS (sliders) ===
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
#### 👩‍💻 Developed by Abhiyanshu Anand and Ishaan Tripathi
""")
    st.markdown("---")



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
# === TOGGLES (remain on main page) ===
color_toggle = st.checkbox("Enable Color Defect Detection", value=True)
deltae_toggle = st.checkbox("Enable DeltaE (LAB) Color Defect Detection", value=True)
pattern_toggle = st.checkbox("Enable Pattern Defect Detection", value=True)
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

def histogram_match(source, reference):
    matched = skimage.exposure.match_histograms(source, reference, channel_axis=-1)
    return matched

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
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
    # Histogram match
    test = histogram_match(test, ref)
    # Denoise
    ref = cv2.fastNlMeansDenoisingColored(ref, None, 10, 10, 7, 21)
    test = cv2.fastNlMeansDenoisingColored(test, None, 10, 10, 7, 21)
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
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    thresh = cv2.threshold(diff, ssim_thresh, 255, cv2.THRESH_BINARY_INV)[1]
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
    return output_img, boxes, defect_percent, diff, thresh

# === SUPERPOINT DEEP KEYPOINT ALIGNMENT ===
def superpoint_align(ref_img, test_img):
    try:
        import torch
        import urllib.request
        import os
        import cv2
        import numpy as np
        # Download SuperPoint weights if not present
        model_url = 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth'
        model_path = 'superpoint_v1.pth'
        if not os.path.exists(model_path):
            with st.spinner('Downloading SuperPoint model...'):
                urllib.request.urlretrieve(model_url, model_path)
        # Minimal SuperPoint model loader (PyTorch)
        class SuperPointNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU(inplace=True)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv1 = torch.nn.Conv2d(1, 64, 3, 1, 1)
                self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
                self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
                self.conv4 = torch.nn.Conv2d(128, 128, 3, 1, 1)
                self.conv5 = torch.nn.Conv2d(128, 256, 3, 1, 1)
                self.conv6 = torch.nn.Conv2d(256, 256, 3, 1, 1)
                self.conv7 = torch.nn.Conv2d(256, 65, 1, 1, 0)
                self.conv8 = torch.nn.Conv2d(256, 256, 1, 1, 0)
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.relu(self.conv4(x))
                x = self.pool(x)
                x = self.relu(self.conv5(x))
                x = self.relu(self.conv6(x))
                cPa = self.conv7(x)
                cDa = self.conv8(x)
                return cPa, cDa
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SuperPointNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        def extract_superpoint_keypoints(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (320, 240))
            timg = torch.from_numpy(img/255.).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                cPa, _ = model(timg)
            prob = torch.nn.functional.softmax(cPa, 1)[0, :-1, :, :]
            prob = prob.cpu().numpy()
            keypoints = np.argwhere(prob > 0.015)
            keypoints = [(float(x[2]*4), float(x[1]*4)) for x in keypoints]
            return keypoints
        kp1 = extract_superpoint_keypoints(ref_img)
        kp2 = extract_superpoint_keypoints(test_img)
        if len(kp1) < 10 or len(kp2) < 10:
            return None, False, "SuperPoint (few keypoints)"
        # Use BFMatcher on keypoints (brute force, nearest neighbor)
        # For simplicity, use Euclidean distance
        matches = []
        for i, pt1 in enumerate(kp1):
            dists = [np.linalg.norm(np.array(pt1)-np.array(pt2)) for pt2 in kp2]
            if len(dists) == 0:
                continue
            min_idx = np.argmin(dists)
            if dists[min_idx] < 50:  # threshold for match
                matches.append((i, min_idx))
        if len(matches) > 10:
            src_pts = np.float32([kp1[i] for i, _ in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[j] for _, j in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            h, w = ref_img.shape[:2]
            aligned_test = cv2.warpPerspective(test_img, M, (w, h))
            alignment_good = len(matches) > 30
            return aligned_test, alignment_good, "SuperPoint"
        return None, False, "SuperPoint (few matches)"
    except Exception as e:
        return None, False, f"SuperPoint error: {e}"

# === ALIGNMENT FUNCTION (deep + classic) ===
def align_images(ref_img, test_img):
    # 1. Try SuperPoint deep keypoint alignment
    aligned_test, alignment_good, method_used = superpoint_align(ref_img, test_img)
    if aligned_test is not None:
        return aligned_test, alignment_good, method_used
    # 2. Try SIFT if available
    try:
        sift = cv2.SIFT_create()
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(test_gray, None)
        if des1 is not None and des2 is not None:
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                h, w = ref_img.shape[:2]
                aligned_test = cv2.warpPerspective(test_img, M, (w, h))
                alignment_good = len(good) > 30
                method_used = "SIFT"
                return aligned_test, alignment_good, method_used
    except Exception as e:
        pass
    # 3. Try ORB as fallback
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
            alignment_good = len(matches) > 30
            method_used = "ORB"
            return aligned_test, alignment_good, method_used
    # 4. Fallback: Multi-scale template matching for zoomed-in test images
    best_val = -1
    best_scale = 1.0
    best_loc = None
    best_size = None
    for scale in np.linspace(0.5, 2.0, 20):
        try:
            resized = cv2.resize(test_gray, (0, 0), fx=scale, fy=scale)
            if resized.shape[0] > ref_gray.shape[0] or resized.shape[1] > ref_gray.shape[1]:
                continue
            res = cv2.matchTemplate(ref_gray, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_scale = scale
                best_loc = max_loc
                best_size = resized.shape
        except Exception as e:
            continue
    if best_val > 0.6 and best_loc is not None:
        x, y = best_loc
        h, w = best_size
        aligned_test = np.zeros_like(ref_img)
        resized_color = cv2.resize(test_img, (w, h))
        aligned_test[y:y+h, x:x+w] = resized_color
        alignment_good = True
        method_used = "TemplateMatching"
        return aligned_test, alignment_good, method_used
    method_used = "None"
    return test_img, False, method_used

def detect_color_defects(ref_img, test_img, threshold=15):
    # Convert to LAB color space
    ref_lab = skcolor.rgb2lab(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    test_lab = skcolor.rgb2lab(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    delta_e = skcolor.deltaE_ciede2000(ref_lab, test_lab)
    mask = delta_e > threshold
    color_defect_img = test_img.copy()
    color_defect_img[mask] = [0, 255, 255]  # Highlight in yellow
    defect_area = np.sum(mask)
    total_area = mask.size
    defect_percent = (defect_area / total_area) * 100
    return color_defect_img, mask, defect_percent, delta_e

def detect_pattern_defects(ref_img, test_img, min_matches=10):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    if des1 is None or des2 is None:
        return test_img, 0, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pattern_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, matches[:min_matches], None, flags=2)
    num_matches = len(matches)
    return pattern_img, num_matches, matches

def auto_rotate_image(test_img, ref_img):
    # Try 0, 90, 180, 270 degree rotations and pick the one with highest SSIM
    best_img = test_img
    best_score = -1
    ref_h, ref_w = ref_img.shape[:2]
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = test_img
        else:
            rotated = cv2.rotate(test_img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        # Resize rotated image to match reference size
        rotated_resized = cv2.resize(rotated, (ref_w, ref_h))
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(rotated_resized, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(ref_gray, test_gray, full=True)
        if score > best_score:
            best_score = score
            best_img = rotated_resized
    return best_img

# === FILE UPLOADS ===
ref_files = st.file_uploader("📁 Upload Reference Images (Multi-Angle)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
test_files = st.file_uploader("🧪 Upload Test Images (Multi-Angle)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
model_file = st.file_uploader("🤖 (Optional) Upload YOLOv8 Model (.pt)", type=["pt"])

# === PROCESSING ===
if ref_files and test_files:
    any_defect = False
    summary_rows = []
    for ref_idx, ref_file in enumerate(ref_files):
        ref_bytes = ref_file.read()
        ref = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        for test_idx, test_file in enumerate(test_files):
            test_bytes = test_file.read()
            test = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)
            st.markdown(f"## 🖼️ Reference {ref_idx+1} vs Test {test_idx+1}")
            if ref is not None and test is not None:
                test = auto_rotate_image(test, ref)
            test_aligned, alignment_good, align_method = align_images(ref, test)
            st.subheader("📸 Uploaded Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(ref, channels="BGR", caption=f"🟢 Reference Image {ref_idx+1}")
            with col2:
                st.image(test, channels="BGR", caption=f"🔍 Test Image {test_idx+1}")
            st.info(f"Alignment method used: {align_method}")
            if not alignment_good:
                st.warning("⚠️ The uploaded images may have very different zoom/scale/orientation or content. Results may not be accurate. Try to upload images with similar field of view and scale.")
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
                # Side-by-side comparison for YOLO
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
            # === SSIM COMPARISON ===
            else:
                st.subheader("🧠 AI Detected Defects (Image Comparison)")
                result_img, detected_boxes, defect_percent, diff_img, mask_img = detect_defects(ref, test_aligned)
                # Side-by-side comparison for SSIM
                st.subheader("📸 Side-by-Side Comparison (SSIM)")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref, channels="BGR", caption=f"🟢 Reference Image {ref_idx+1}")
                with col2:
                    st.image(result_img, channels="BGR", caption="🔴 Defective Image (Differences Highlighted)")
                st.image(diff_img, caption="SSIM Difference Image", clamp=True)
                st.image(mask_img, caption="Defect Mask", clamp=True)
                if defect_percent > 0.5:
                    st.error(f"⚠️ Defects Found: {defect_percent:.2f}% of the area")
                    any_defect = True
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "SSIM", "Defect": True})
                else:
                    st.success("✅ No major defects detected 🎈")
                    summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "SSIM", "Defect": False})
                if detected_boxes:
                    st.subheader("📋 Defect Table")
                    df = pd.DataFrame(detected_boxes)
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode()
                    st.download_button(f"⬇️ Download Defect Report (CSV) - Ref{ref_idx+1}_Test{test_idx+1}", csv, f"defect_report_ref{ref_idx+1}_test{test_idx+1}.csv", "text/csv")
                # === Color Defect Detection ===
                if color_toggle and deltae_toggle:
                    st.subheader("🎨 Color Defect Detection (Delta E 2000)")
                    color_img, color_mask, color_defect_percent, delta_e_img = detect_color_defects(ref, test_aligned, threshold=color_thresh)
                    st.image(color_img, channels="BGR", caption="🟡 Color Differences Highlighted")
                    st.image(delta_e_img, caption="Delta E Map", clamp=True)
                    if color_defect_percent > 0.5:
                        st.error(f"⚠️ Color Defects Found: {color_defect_percent:.2f}% of the area")
                        any_defect = True
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Color", "Defect": True})
                    else:
                        st.success("✅ No major color defects detected 🎈")
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Color", "Defect": False})
                # === Pattern Defect Detection ===
                if pattern_toggle:
                    st.subheader("🔳 Pattern Defect Detection (ORB)")
                    pattern_img, num_matches, matches = detect_pattern_defects(ref, test_aligned, min_matches=pattern_min_matches)
                    st.image(pattern_img, channels="BGR", caption=f"Pattern Matches: {num_matches}")
                    if num_matches < pattern_min_matches:
                        st.error("⚠️ Pattern mismatch detected!")
                        any_defect = True
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Pattern", "Defect": True})
                    else:
                        st.success("✅ Pattern matches are sufficient 🎈")
                        summary_rows.append({"Reference": ref_idx+1, "Test": test_idx+1, "Type": "Pattern", "Defect": False})
    # Summary for all images
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
