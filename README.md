# HAL Aircraft Part Side Detector

A computer vision project for Hindustan Aeronautics Limited (HAL) Summer Internship.  
Detects aircraft hardware parts (like side mirrors, side doors, landing gears, etc.) and classifies their orientation (Left or Right) using YOLOv8 and a Streamlit web app.

---

## 🚀 Features

- **YOLOv8-based Detection:** Detects and classifies aircraft parts and their side (Left/Right).
- **Streamlit Web App:** User-friendly interface for image upload and result visualization.
- **Custom Dataset Support:** Upload your own YOLOv8-format dataset and retrain the model directly from the app.
- **Downloadable Results:** Export detection results as CSV.

---

## 🗂️ Project Structure

```
HAL-Internship-Project/
│
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── hal_logo.png                # HAL logo for branding
└── ...
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhi110704/HAL-Internship-Project.git
   cd HAL-Internship-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download or prepare your YOLOv8 dataset and weights.**
   - Place your dataset and weights in the appropriate folders as shown above.

---

## 🏃‍♂️ Usage

1. **Train the Model (Optional):**
   - If you want to retrain on your own dataset:
     ```bash
     python train_model.py
     ```

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser and go to:**  
   [http://localhost:8501](http://localhost:8501)

4. **Upload images or use the HAL dataset to see detection results.**

---

## 📦 Custom Dataset Upload

- Use the sidebar in the app to upload a YOLOv8-format dataset (.zip).
- The app will retrain the model and use the new weights for detection.

---

## 📄 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Streamlit
- Pillow
- pandas
- numpy

(See `requirements.txt` for details.)

---

## 🙏 Acknowledgements

- Hindustan Aeronautics Limited (HAL)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Streamlit

---

## 👨‍💻 Developed by

**Abhiyanshu Anand**  
[GitHub](https://github.com/Abhi110704)

---

**For any issues or suggestions, please open an issue on this repository.** 