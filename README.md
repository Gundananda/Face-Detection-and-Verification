# 👤 Face Detection and Verification with MTCNN + FaceNet

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow/Keras](https://img.shields.io/badge/Keras-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MTCNN](https://img.shields.io/badge/MTCNN-Face%20Detection-8A2BE2)](https://github.com/ipazc/mtcnn)
[![FaceNet](https://img.shields.io/badge/FaceNet-Embeddings-D00000)](https://github.com/nyoki-mtl/keras-facenet)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end‑to‑end notebook that performs Face Detection with MTCNN and Face Verification using FaceNet embeddings with cosine‑similarity scoring, including evaluation, ROC, and threshold tuning.

</div>

---

## 📌 Overview

This project demonstrates:
- Face Detection: detect faces and facial keypoints using MTCNN.
- Face Verification: verify whether two faces belong to the same person using FaceNet embeddings and cosine similarity.
- Evaluation: generate confusion matrix, ROC curve, metric report, and similarity score distributions to choose an operating threshold.

It uses pre‑trained models (no training required), works with local images or URLs, and includes utility functions to streamline the full pipeline.

---

## ✨ Key Features

- MTCNN face detection with bounding boxes and five facial keypoints.
- Face cropping/alignment and resizing to 160×160 for FaceNet.
- FaceNet embeddings and cosine similarity for one‑to‑one verification.
- Thresholdable decision rule (default 0.7); ROC/AUC to tune threshold.
- Plots: accuracy/loss (if training variants used), confusion matrix, ROC, and similarity histograms.
- Modular utility functions for detection, embedding, and comparison.

---

## 📂 Project Structure

```plaintext
face-verification-mtcnn-facenet/
├── FaceDetectionVerification.ipynb   # Main notebook (detection, verification, evaluation)
├── archive.zip                       # Dataset zip (example) - not in repo by default
├── README.md
└── LICENSE
```

Note: The sample notebook expects a zip like archive.zip with a folder such as Football players/train/<person>/*.jpg. Replace with your own dataset and update paths accordingly.

---

## 🧰 Requirements

- Python 3.9+
- TensorFlow/Keras, OpenCV, MTCNN, keras-facenet, scikit‑learn, seaborn, matplotlib

Install:
```bash
pip install mtcnn keras-facenet opencv-python tensorflow scikit-learn seaborn matplotlib
```

Optional (for URL image loading convenience):
```bash
pip install scikit-image
```

---

## 🚀 Quickstart

1) Place or unzip your dataset
- Example in the notebook uses archive.zip with subfolders per identity.
- After unzip, update paths like /content/Football players/train/<name>/<img>.jpg.

2) Open the notebook and run cells in order
- Install packages
- Import libraries
- Extract/unzip dataset (if needed)

3) Face detection demo
- Run Face_Detection(image=..., url=...) to draw boxes and keypoints using MTCNN.

4) Face verification
- Functions:
  - reading_img(path) → RGB image
  - Face_Detection_FaceNet(image, margin=0.2) → aligned face crop (160×160)
  - Embeding(face_crop) → 512‑D FaceNet embedding
  - Similarity_Measurement(emb1, emb2, threshold=0.7) → True/False
  - Comparison_Faces(img1, img2) → show crops + decision
  - showing(image_path1, image_path2) → prints a friendly result

5) Evaluation
- Build test_pairs of (img1, img2) with true_labels (1 = same, 0 = different).
- Compute cosine similarities, predicted labels with a chosen threshold.
- Report:
  - Accuracy, precision, recall, F1 (classification_report)
  - Confusion matrix
  - ROC curve + AUC
  - Similarity score histograms for “same” vs “different”

6) Tune threshold
- Use the ROC curve / Youden’s J (tpr − fpr) to select a threshold that balances FPR/TPR for your use case.

---

## 🧠 How It Works

- MTCNN (Multi‑Task Cascaded Convolutional Networks) detects faces and returns bounding boxes + keypoints.
- We crop and resize faces (160×160), as expected by FaceNet.
- FaceNet produces embeddings (feature vectors) for each face.
- Cosine similarity between two embeddings measures how close two faces are.
- If similarity ≥ threshold → same person; otherwise different.

---

## 📝 Tips

- Alignment/margin: Adjust margin in Face_Detection_FaceNet for tighter/looser crops.
- Lighting/pose: Preprocess (denoise, histogram equalization) if your images vary widely.
- Thresholds: 0.5–0.8 are common starting points; validate on your own pairs.
- Multiple faces: The demo uses the first detected face; adapt if images contain multiple faces.

---

## ⚖️ Limitations & Ethics

- This is a verification demo, not a production system.
- Performance depends on imaging conditions (pose, occlusion, lighting).
- Always obtain consent and follow local laws and platform policies when processing faces.
- Do not use on images without appropriate rights. Replace sample data with your own images.

---

## 🔧 Troubleshooting

- No face detected: check image quality/size; adjust margin; try a different detector configuration.
- Wrong matches: revisit threshold; ensure proper cropping/alignment; verify identity labels for test pairs.
- Performance issues: batch embedding calls where possible; reduce image size for detection preview.

---

## 📄 License

Released under the MIT License. See LICENSE.

---


⭐️ If this helped, consider giving the repo a star!
