# 🛠️ HAL Parts Defect Detection System

A government-grade web application for AI-powered defect detection in aircraft parts, developed for Hindustan Aeronautics Limited (HAL).

## 🚀 Introduction
This project provides a user-friendly interface for detecting defects in aircraft parts using both image comparison and deep learning (YOLOv8). It is designed for use by HAL and other aerospace organizations to ensure quality and safety in manufacturing and maintenance.

## ✨ Features
- Upload reference and test images for defect analysis
- Optional upload of custom YOLOv8 model (.pt) for deep learning-based detection
- Visual and tabular defect reports
- Downloadable CSV reports
- Government-compliant UI and disclaimers

## 📦 Requirements
See [`requirement.txt`](requirement.txt) for all dependencies.

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hal-defect-detection.git
   cd hal-defect-detection/Defect Identify
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
4. Place `hal_logo.png` in the same folder as `M.py
5. Run the app:
   ```bash
   streamlit run M.py
   ```

## 📝 Usage
1. Upload a reference image (ideal, defect-free part).
2. Upload a test image (part to inspect).
3. (Optional) Upload a YOLOv8 `.pt` model for advanced detection.
4. View results, download reports, and ensure quality!

## 👩‍💻 Developers
- Abhiyanshi Anand
- Ishaan Tripathi

## 🏛️ About HAL
Hindustan Aeronautics Limited (HAL) is an Indian state-owned aerospace and defence company involved in the design, fabrication, and assembly of aircraft, jet engines, helicopters, and their spare parts.

## ⚠️ Disclaimer
This is an official government application. Unauthorized access or misuse is prohibited and may be punishable under applicable laws. Data uploaded is used only for defect detection purposes.

## 📄 License
This project is for educational and internal use at HAL. For other use, please contact the developers or HAL.

**Note:** For best results, upload images with similar orientation and zoom (scale). The system will try to auto-align and scale images, but similar field of view improves accuracy and speed.

## 🧠 Approach & Algorithms

### 1. Image Upload & Preprocessing
- Users upload a **reference image** (defect-free) and a **test image** (to inspect).
- Optionally, a custom YOLOv8 model can be uploaded for deep learning-based detection.

### 2. Image Alignment (Rotation & Zoom Correction)
- **Goal:** Ensure the test image matches the orientation and scale of the reference image.
- **Algorithm:**
  - Uses **ORB (Oriented FAST and Rotated BRIEF)** feature detection to find keypoints and descriptors in both images.
  - Matches features using **Brute-Force Matcher** with Hamming distance.
  - If enough matches are found, computes a **homography matrix** using RANSAC to estimate the geometric transformation (rotation, scaling, translation) between the test and reference images.
  - Applies **cv2.warpPerspective** to align the test image to the reference image.
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

### 4. User Guidance & Robustness
- The system automatically tries to correct for rotation and zoom differences.
- Users are advised to upload images with similar orientation and zoom for best results.
- If alignment is poor, a warning is displayed.

### 5. Reporting
- Detected defects (from either method) are shown in a table.
- Users can download the results as a CSV report.

### Summary Table

| Step                | Algorithm/Method         | Library/Tool         |
|---------------------|-------------------------|----------------------|
| Alignment           | ORB, BFMatcher, Homography | OpenCV              |
| Image Comparison    | SSIM, Thresholding, Contours | scikit-image, OpenCV|
| Deep Learning       | YOLOv8                   | ultralytics          |
| Reporting           | DataFrame, CSV           | pandas               | 