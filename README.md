# 🛠️ HAL Parts Defect Detection System

A government-grade web application for AI-powered defect detection in aircraft parts, developed for Hindustan Aeronautics Limited (HAL).

## 🚀 Introduction
This project provides a user-friendly interface for detecting defects in aircraft parts using both image comparison and deep learning (YOLOv8). It is designed for use by HAL and other aerospace organizations to ensure quality and safety in manufacturing and maintenance.

## ✨ Features
- Upload **multiple reference** and **multiple test images** for multi-angle, multi-reference analysis
- Optional upload of custom YOLOv8 model (.pt) for deep learning-based detection
- Visual and tabular defect reports for every (reference, test) pair
- Side-by-side comparison of reference and defective (highlighted) image for both SSIM and YOLO detection, making visual inspection easier
- Downloadable CSV reports (per pair and summary)
- Sidebar controls for detection sensitivity (SSIM, color, pattern, YOLO confidence)
- Toggle buttons for enabling/disabling color, DeltaE (LAB), and pattern defect detection
- Deep learning-based image alignment (SuperPoint, SIFT, ORB, template matching)
- Government-compliant UI and disclaimers

## 📦 Requirements
See [`requirement.txt`](requirement.txt) for all dependencies.
- **torch** is required for SuperPoint deep alignment (auto-downloads model on first run)

## ⚙️ Installation (Version-wise Guide)

### 1. **Python Version**
- **Recommended:** Python 3.8 – 3.11
- Python 3.12+ may work, but some libraries (torch, opencv-python) may not have wheels yet. If you get install errors, use Python 3.10 or 3.11.

### 2. **Create a Virtual Environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirement.txt
```

### 4. **Check Python & Library Versions**
```bash
python --version
pip show streamlit opencv-python torch ultralytics scikit-image
```
- Make sure Python is 3.8–3.11 and all libraries are installed without errors.

### 5. **Place Required Files**
- Place `hal_logo.png` in the same folder as `M.py`.

### 6. **Run the App**
You can use either of the following commands:
```bash
streamlit run M.py
# or
python -m streamlit run M.py
```

### 7. **First Run (SuperPoint Model Download)**
- On first run, the SuperPoint model (`superpoint_v1.pth`) will be downloaded automatically if not present.

---

## 📝 Usage
1. Upload one or more reference images (ideal, defect-free part).
2. Upload one or more test images (parts to inspect, from any angle/zoom).
3. (Optional) Upload a YOLOv8 `.pt` model for advanced detection.
4. Adjust detection sensitivity in the sidebar.
5. Use toggles on the main page to enable/disable color, DeltaE, and pattern detection.
6. View side-by-side comparison of reference and defective (highlighted) image for both SSIM and YOLO detection.
7. View results, download reports, and ensure quality!

## 👩‍💻 Developers
- Abhiyanshu Anand
- Ishaan Tripathi

## 🏛️ About HAL
Hindustan Aeronautics Limited (HAL) is an Indian state-owned aerospace and defence company involved in the design, fabrication, and assembly of aircraft, jet engines, helicopters, and their spare parts.

## ⚠️ Disclaimer
This is an official government application. Unauthorized access or misuse is prohibited and may be punishable under applicable laws. Data uploaded is used only for defect detection purposes.

## 📄 License
This project is for educational and internal use at HAL. For other use, please contact the developers or HAL.

**Note:** For best results, upload images with similar orientation and zoom (scale). The system will try to auto-align and scale images, but similar field of view improves accuracy and speed. Deep alignment (SuperPoint) is used for robust matching, but extreme cases may still require careful image capture.

## 🧠 Approach & Algorithms

### 1. Image Upload & Preprocessing
- Users upload one or more **reference images** (defect-free) and one or more **test images** (to inspect).
- Optionally, a custom YOLOv8 model can be uploaded for deep learning-based detection.

### 2. Image Alignment (Rotation, Zoom, and Deep Correction)
- **Goal:** Ensure the test image matches the orientation and scale of the reference image, even under zoom, crop, or rotation.
- **Algorithm:**
  - Tries **SuperPoint** (deep learning keypoint detector) for robust alignment under extreme zoom, rotation, and translation.
  - Falls back to **SIFT**, then **ORB**, then multi-scale template matching if needed.
  - Shows which method was used for each image pair in the UI.
  - If alignment is poor (few matches), a warning is shown to the user.

### 3. Defect Detection Methods
#### a. SSIM-Based Image Comparison
- **SSIM (Structural Similarity Index):**
  - Compares the aligned test image to the reference image pixel-by-pixel.
  - Highlights regions with significant differences, which are likely defects.
  - Uses adaptive thresholding and contour detection to localize and classify defects (e.g., scratch, dent, crack) based on area and aspect ratio.

#### b. YOLOv8 Deep Learning Detection (Optional)
- If a YOLOv8 model is provided:
  - The model is loaded and run on the test image.
  - Detected defects are shown with bounding boxes, labels, and confidence scores.
  - Results are presented visually and in a downloadable CSV report.

#### c. Color Defect Detection (DeltaE, LAB)
- Toggleable DeltaE (LAB) color difference detection for subtle color changes.
- Adjustable threshold in the sidebar.

#### d. Pattern Defect Detection (ORB)
- Toggleable pattern matching for structural differences.
- Adjustable minimum match threshold in the sidebar.

### 4. User Guidance & Robustness
- The system automatically tries to correct for rotation, zoom, and crop differences using deep and classic methods.
- Users are advised to upload images with similar orientation and zoom for best results.
- If alignment is poor, a warning is displayed.

### 5. Reporting
- Detected defects (from any method) are shown in a table.
- Users can download the results as a CSV report (per pair and summary).

### Summary Table

| Step                | Algorithm/Method         | Library/Tool         |
|---------------------|-------------------------|----------------------|
| Alignment           | SuperPoint, SIFT, ORB, Template Matching | torch, OpenCV |
| Image Comparison    | SSIM, Thresholding, Contours | scikit-image, OpenCV|
| Deep Learning       | YOLOv8                   | ultralytics          |
| Reporting           | DataFrame, CSV           | pandas               | 
