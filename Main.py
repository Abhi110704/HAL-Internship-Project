import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64
import tempfile
from datetime import datetime
from fpdf import FPDF
from skimage.metrics import structural_similarity as ssim

HAL_LOGO_PATH = "hal_logo.png"
THRESHOLD = 0.90

st.set_page_config(page_title="HAL Aircraft Part Defect Detector", layout="wide")

with st.sidebar:
    st.image(HAL_LOGO_PATH, caption="HAL Logo", use_container_width=True)
    st.title("🔧 Navigation")
    st.markdown("""
    - 📄 [Instructions](#📋-how-to-use-this-tool)
    - 🖼️ Upload Images
    - 🔍 View Results
    - 📥 Download Report
    """)
    st.markdown("---")
    st.markdown("Developed by Abhiyanshu Anand and Ishaan Tripathi")

st.markdown("<h1 style='text-align: center;'>🔍 HAL (Hindustan Aeronautics Limited) Aircraft Part Defect Detector</h1>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
### 📋 How to Use This Tool:

1. **Upload Images**:
   - Upload a clear **Reference Image** (ideal/defect-free image).
   - Upload the **Test Image** (image of the part to be tested).

2. **Defect Detection**:
   - The tool will compare both images and detect structural differences.
   - It classifies the defect type into **Scratch**, **Dent**, **Crack**, or **Shape Mismatch**.
   - Defective regions are highlighted in green on the test image.

3. **Center of Mass**:
   - Coordinates of center of mass will be shown for the detected shape.

4. **SSIM Score**:
   - A Structural Similarity Index (SSIM) score between 0 and 1 is calculated.
   - A score **≥ 0.90** indicates no significant defects.
   - A score **< 0.90** indicates visible defects.

5. **View Results**:
   - You'll see the SSIM score, number of defects, and highlighted regions.
   - A message will indicate whether the part passed or failed the inspection.

6. **Download Report**:
   - A **PDF report** will be generated summarizing:
     - Time of inspection
     - SSIM score
     - Number and type of defects
     - Defect severity level
   - Click on the link to **Download Report**.

---
### 🖼️ Upload Reference and Test Images
""")

col1, col2 = st.columns(2)

with col1:
    ref_img_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref")
with col2:
    test_img_file = st.file_uploader("Upload Image to Test", type=["png", "jpg", "jpeg"], key="test")

def load_image(image_file):
    return np.array(Image.open(image_file).convert("RGB"))

def get_defect_boxes(diff_img):
    contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
    return boxes

def classify_defect(region):
    mean_intensity = np.mean(region)
    if mean_intensity > 180:
        return "Scratch"
    elif mean_intensity > 100:
        return "Dent"
    elif mean_intensity > 30:
        return "Crack"
    else:
        return "Shape Mismatch"

def calculate_severity(area):
    if area > 3000:
        return "High"
    elif area > 1500:
        return "Moderate"
    else:
        return "Low"

def calculate_center_of_mass(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def draw_dotted_rectangle(img, pt1, pt2, color, thickness=2, gap=5):
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw top and bottom dotted lines
    for i in range(x1, x2, gap*2):
        cv2.line(img, (i, y1), (min(i+gap, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i+gap, x2), y2), color, thickness)
    # Draw left and right dotted lines
    for i in range(y1, y2, gap*2):
        cv2.line(img, (x1, i), (x1, min(i+gap, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i+gap, y2)), color, thickness)

def get_part_outline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    outline = np.zeros_like(image)
    # Draw edges in white
    outline[edges > 0] = [255, 255, 255]
    return outline

def compare_images(ref_img, test_img):
    ref_img = cv2.resize(ref_img, (256, 256))
    test_img = cv2.resize(test_img, (256, 256))

    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

    score, diff = ssim(gray_ref, gray_test, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
    boxes = get_defect_boxes(thresh)
    classified_boxes = []

    # Get outline image
    outline_img = get_part_outline(test_img)
    overlay = outline_img.copy()

    cx, cy = calculate_center_of_mass(test_img)
    cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
    cv2.putText(overlay, f"Center: ({cx}, {cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    known_types = ["Scratch", "Dent", "Crack", "Shape Mismatch"]

    for idx, (x, y, w, h) in enumerate(boxes):
        defect_roi = test_img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(defect_roi, cv2.COLOR_RGB2GRAY)
        defect_type = classify_defect(gray_roi)
        severity = calculate_severity(w * h)
        color = (0, 255, 0) if defect_type in known_types else (255, 0, 255)  # Green for known, Pink for new
        # Draw dotted rectangle
        draw_dotted_rectangle(overlay, (x, y), (x + w, y + h), color, thickness=2, gap=6)
        # Mark center of defect
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(overlay, (center_x, center_y), 4, (0, 0, 255), -1)
        # Add label
        cv2.putText(overlay, f"Defect {idx+1}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(overlay, f"{defect_type} ({severity})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        classified_boxes.append((x, y, w, h, defect_type, severity))

    mismatch = round((1 - score) * 100, 1)
    return score, overlay, classified_boxes, mismatch, (cx, cy), test_img, boxes

def create_pdf_report(score, classified_boxes, mismatch, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="HAL Aircraft Part Defect Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"SSIM Score: {score:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Defect Detected: {'Yes' if score < THRESHOLD else 'No'}", ln=True)
    pdf.cell(200, 10, txt=f"Mismatch Percentage: {mismatch}%", ln=True)
    pdf.cell(200, 10, txt=f"Number of Defective Regions: {len(classified_boxes)}", ln=True)
    for i, (x, y, w, h, defect_type, severity) in enumerate(classified_boxes):
        pdf.cell(200, 10, txt=f"Region {i+1}: {defect_type} ({severity}) at ({x}, {y}, {w}, {h})", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{filename}">📄 Download Report</a>'
    return href

if ref_img_file and test_img_file:
    ref_img = load_image(ref_img_file)
    test_img = load_image(test_img_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_img, caption="Reference Image", use_container_width=True)
    with col2:
        st.image(test_img, caption="Test Image", use_container_width=True)

    with st.spinner("🔍 Comparing..."):
        score, overlay, classified_boxes, mismatch, (cx, cy), test_img_arr, defect_boxes = compare_images(ref_img, test_img)
        # Also get reference outline for comparison
        ref_outline = get_part_outline(ref_img)
        # Draw defect regions on reference outline as well
        ref_outline_with_boxes = ref_outline.copy()
        for idx, (x, y, w, h, defect_type, severity) in enumerate(classified_boxes):
            color = (0, 255, 255)  # Yellow for reference highlight
            draw_dotted_rectangle(ref_outline_with_boxes, (x, y), (x + w, y + h), color, thickness=2, gap=6)
            # Optionally, add label
            cv2.putText(ref_outline_with_boxes, f"Defect {idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    st.markdown(f"### 🧠 SSIM Score: **{score:.2f}**")
    if score >= THRESHOLD:
        st.success("✅ No major defects detected. Parts match closely.")
        st.balloons()
    else:
        st.error("❌ Defect Detected!")
        st.markdown(f"**Defect Severity:** {mismatch}% mismatch")
        # Side-by-side comparison
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.image(ref_outline_with_boxes, caption="Reference Outline with Defect Regions", width=328)
        with comp_col2:
            st.image(overlay, caption="Test Outline with Defect Highlighted", width=328)
        st.markdown(f"🔳 Number of Defective Regions: **{len(classified_boxes)}**")
        # Calculate and display average defect size
        if classified_boxes:
            areas = [w * h for (_, _, w, h, _, _) in classified_boxes]
            avg_area = sum(areas) / len(areas)
            st.markdown(f"📏 **Average Defect Size:** {avg_area:.1f} pixels²")
            # Show average defect region as thumbnail
            avg_idx = np.argmin([abs((w*h) - avg_area) for (_, _, w, h, _, _) in classified_boxes])
            x, y, w, h, _, _ = classified_boxes[avg_idx]
            defect_crop = test_img_arr[y:y+h, x:x+w]
            thumb = cv2.resize(defect_crop, (min(96, w), min(96, h)), interpolation=cv2.INTER_AREA)
            st.image(thumb, caption="Average Defect Region (thumbnail)", width=96)
            # Show all defect regions and their locations
            st.markdown("#### 🗺️ Defective Regions:")
            for i, (x, y, w, h, defect_type, severity) in enumerate(classified_boxes):
                st.markdown(f"- **Region {i+1}:** {defect_type} ({severity}) at **({x}, {y}, {w}, {h})**")
        st.markdown(f"📍 Center of Mass: **({cx}, {cy})**")

        selected_types = st.multiselect("🔍 Filter by defect types", list(set([t for *_, t, _ in classified_boxes])))
        for i, (x, y, w, h, defect_type, severity) in enumerate(classified_boxes):
            if not selected_types or defect_type in selected_types:
                st.markdown(f"🔸 Region {i+1}: **{defect_type}** ({severity}) at **({x}, {y}, {w}, {h})**")

        if mismatch > 40:
            st.warning("⚠️ High level of defect detected!")
        else:
            st.info("🔎 Minor differences detected.")

    st.markdown("---")
    report_link = create_pdf_report(score, classified_boxes, mismatch, "hal_defect_report.pdf")
    st.markdown(report_link, unsafe_allow_html=True)

st.markdown("---")
st.info("Upload reference and test images to compare and detect defects. The system highlights mismatches and classifies defect types.")
