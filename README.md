# AI-Powered Document Scanner with OCR

This project is a software-based document scanner built using **Python, OpenCV, and Tesseract OCR**.
It automatically detects document boundaries, enhances image quality, corrects perspective distortions,
and extracts text from printed or handwritten documents. The system supports **English, Hindi, and Gujarati**
and performs well on both clean and noisy images.

For a full technical explanation, methodology, and results, please refer to the [Project_Report.pdf](Project_Report.pdf) included in this repository.

---

## Features

- Automatic and manual document boundary detection
- Image preprocessing (CLAHE, denoising, thresholding, morphological cleanup)
- Perspective correction
- OCR using Tesseract
- Spell correction for English text
- Export results as:
  - Preprocessed image
  - OCR-ready image
  - High-quality JPG
  - PDF
  - Extracted text file

---

## Project Workflow

1. **Image Acquisition**
2. **Image Preprocessing**
3. **Document Boundary Detection**
4. **Perspective Transformation**
5. **Optical Character Recognition (OCR)**
6. **Post-Processing**
7. **Output Generation**

For detailed explanations, example outputs, and accuracy comparisons, see:  
**Project_Report.pdf** (root folder)

---

## Installation

Install Python3 from https://www.python.org/downloads/

Clone repository:

```
git clone https://github.com/krushangptl/Doc-Scanner-Project
cd Doc-Scanner-Project
```

Install requirements:

```
pip install -r requirements.txt
```

Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract

## How to Run

### 1. Load Program

Open `main.ipynb` inside your VS Code or Jupyter-lab or Google Colab.

### 2. Set the image path inside your script

```
IMG = "path/to/your/image.jpg"
```

### 3. Select Options in console:

- Preprocessing
- Thresholding
- OCR Language

### 4. Output

Check out /out for your output images and extracted text from image.

## Supported Language

- English
- Gujarati
- Hindi

Note: Spell correction is applied only for English.

## Overall Evaluation for All Language-Specific Documents

### Confidence Scoring

| Document Type        | Original (%) | Noise Image (%) | Normal Thresholding (%) |
| -------------------- | ------------ | --------------- | ----------------------- |
| Gujarati Handwritten | 69.2         | 45.1            | 76.0                    |
| Hindi Handwritten    | 61.4         | -               | No Text Detected        |
| English Handwritten  | 47.9         | -               | 68.3                    |
| Gujarati Printed     | 94.6         | 92.6            | 86.2                    |
| Hindi Printed        | 90.5         | 95.0            | 95.0                    |
| English Printed      | 95.5         | 92.0            | 95.3                    |
