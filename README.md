
# Deep Learning-based Anomaly Detection

## 🧠 Project Overview

This research project explores the application of deep learning models, particularly **Autoencoders**, **Multi-Layer Perceptrons (MLP)**, and **Deep Q-Learning (DQL)**, for anomaly detection. The models are benchmarked against classical unsupervised methods such as **Isolation Forest**, and **Local Outlier Factor**

The project includes:
- Evaluation on Scikit-learn datasets (e.g., blobs, moons)
- Application to the **Kaggle Credit Card Fraud Detection** dataset
- Custom implementation of a Deep Q-Learning anomaly detection model
- Comparison of macro F1-score across models and datasets
- Visualization of decision boundaries

---

## 🗂️ Project Structure

```
project_root/
│── data/
│   ├── raw/                # Original datasets (Credit Card Fraud)
│
│── notebooks/              # Jupyter notebooks for experiments
│   ├── 01_isolation_forest_anomaly_detection.ipynb
│   ├── 02_compare_models_Is_local.ipynb
│   ├── 03_mlp_classifier.ipynb
│   ├── 04_autoencoder_pipeline.ipynb
│   ├── 05_autoencoder_pipeline_credit_card_fraud_detection.ipynb
│   ├── 06_model_comparison_full.ipynb
│   ├── 07_deep_autoencoder_visualization.ipynb
│   └── comparison_with_real_dql.ipynb
│
│── models/                 # Saved trained models
│── scripts/                # Python scripts (real_dql_model.py, utils.py)
│── requirements.txt        # Python dependencies
│── README.md               # Project overview
```

---

## 📊 Dataset: Credit Card Fraud Detection

We use the **Credit Card Fraud Detection** dataset from Kaggle:

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Due to GitHub's 100MB file limit, the dataset is not included.  
After downloading, place it here:

```
Notebook/data/raw/creditcard.csv
```

---

## ⚙️ Models Used

- **Autoencoder**: Deep symmetric architecture with dropout and regularization
- **MLP Classifier**: Simple feedforward model for balanced data comparison
- **Deep Q-Learning**: Custom agent-based anomaly detector
- **Classical Models**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - SGDOneClassSVM

---

## 📈 Evaluation Metrics

- **Macro Average F1-Score**
- **Classification Report**
- **Reconstruction Error (MSE)**
- **Decision Boundary Visualizations**

---

## 🧪 How to Run

Make sure you have the required libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

Run each notebook individually depending on the model and dataset.

---

## 🛡️ Disclaimer

This repository is intended for **academic and research purposes** only.  
Do not reuse without proper citation of:
- The Kaggle dataset
- TensorFlow/Keras
- scikit-learn
- Any third-party GitHub or Kaggle tutorials referenced in headers

---

## 📬 Contact

For questions or contributions, please open an issue or contact:

**Ammar Abrahani**  
[GitHub](https://github.com/ammarabrahani/deep-learning-anomaly-k8s)

