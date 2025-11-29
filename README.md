# 🌍 Major Quake Recall Enhancement  
### Evaluating ML Model Performance Using a GenAI-Synthesized Dataset  
### A Case Study in Earthquake Prediction

This project applies **Generative AI (CTGAN)** and **Machine Learning (Random Forest)** to improve the recall of **major earthquakes** in highly imbalanced seismic datasets. The work is based on the research study documented in the paper *"Evaluating ML Model Performance Using a GenAI-Synthesized Dataset: A Case Study in Earthquake Prediction"* :contentReference[oaicite:1]{index=1}.

---

## 📌 Problem Overview

Earthquake datasets suffer from **extreme class imbalance** — minor quakes occur very frequently, while **major quakes (>5.5)** are rare.  
Most ML models obtain high accuracy but **fail to detect major earthquakes**, which are the most critical for disaster prediction and early-warning systems.  

This project answers:

- Can CTGAN generate realistic synthetic major-earthquake samples?  
- Does augmentation improve recall, precision, and F1-score?  
- Can a Random Forest trained on synthetic data outperform the baseline model?

---

## 🚀 Key Contributions

✔ Built an augmentation-first pipeline using **CTGAN**  
✔ Generated realistic synthetic samples for the minority (major quake) class  
✔ Trained & compared two Random Forest classifiers:  
  - **Baseline Model** → trained on original data  
  - **Enhanced Model** → trained on CTGAN-augmented data  
✔ Achieved **2× improvement** in major-quake recall  
✔ Maintained high overall accuracy (>97%) even after augmentation  

---

## 📁 Dataset Summary

Source: **USGS Earthquake Catalog**  
- 19,824 earthquake records  
- Features include: Latitude, Longitude, Depth, Magnitude Type, NST, GAP, RMS  
- magType encoded using one-hot encoding  
- Binary target created as:
  is_major_quake = 1 if magnitude > 5.5 else 0


The dataset is **heavily imbalanced** → only **1.72%** are major earthquakes.

---

## 🧠 Methodology Overview

### **1. Data Preprocessing**
- Removed missing/invalid values  
- One-hot encoded the magnitude type  
- Removed raw magnitude to avoid label leakage  

### **2. Train–Test Split**
- 80–20 stratified split  
- Test set: 3,703 minor & 65 major quakes  

### **3. CTGAN Training**
- Mode-specific normalization  
- Conditional sampling on discrete columns  
- Trained for 300 epochs  
- Generated 50,000 synthetic samples  

### **4. Dataset Augmentation**
- Extracted synthetic major-quake samples  
- Balanced the dataset by adding required synthetic samples  

### **5. Model Training**
Two Random Forests were trained:

- **Baseline Model** → Real data only  
- **Enhanced Model** → Real + Synthetic data  

---

## 📊 Results Summary

### **Baseline Model (Imbalanced Data)**
- **Recall (Major)**: 26.15%  
- Correct major-quake predictions: **17 / 65**  
- Accuracy: 98.43%  

### **Enhanced Model (CTGAN-Augmented Data)**
- **Recall (Major)**: 58.46%  
- Correct major-quake predictions: **38 / 65**  
- Accuracy: 97.37%  

➡️ **Major-quake recall increased from 26% → 58%**, more than **2× improvement**.

False positives increased — but this is acceptable because **missing a major quake is worse than a false alarm**.

---

## 📌 Visualizations Included

- Class distribution  
- Recall comparison (baseline vs enhanced)  
- Feature importance  
- Earthquake geographic map  
- Confusion matrix  

---

## 🛠 Tools & Technologies

- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- SDV (CTGAN)  
- Matplotlib / Plotly  
- Streamlit  
- Joblib  

---

## ▶️ How to Run the Project

### **1. Create virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Start the Streamlit app**
```
streamlit run app.py
```


---

## 🧩 Project Structure

```
├── app.py
├── data/
├── models/
├── utils/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🏁 Conclusion

CTGAN-based augmentation significantly strengthens ML model performance in rare-event earthquake prediction.  
The enhanced Random Forest **doubled recall** for major earthquakes while maintaining strong accuracy.

This approach is practical for real-world seismic monitoring and can be expanded with:
- CopulaGAN or TVAE  
- Temporal data (foreshocks/aftershocks)  
- Deep learning on raw waveform data  
- Real-time deployment  

---

## 👤 Authors

- **Hemakesh G , Kishore K, Kishan Shree B**  
- Mentors: Dr. G. Bhuvaneswari, Arul Dalton G, Madhumitha N  
- Saveetha Engineering College  

GitHub: https://github.com/HEMAKESHG  
LinkedIn: https://linkedin.com/in/hemakesh-g-714745285


