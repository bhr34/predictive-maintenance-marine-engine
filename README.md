# Marine Engine Preventive Maintenance – Risk Classification

This project analyzes marine engine sensor data to classify engine status as either **"normal"** or **"at risk"** using machine learning.  
It aims to support **predictive maintenance** strategies and reduce unplanned failures in marine systems.

---

## 🔍 Objective

To build a classification model that identifies whether an engine is at high risk of failure **based on sensor inputs**, so that preventive maintenance can be scheduled in time.

> 🛠 Threshold logic: An engine temperature over **95°C** is considered at risk.

---

## 📁 Project Structure


---

## 🧠 What I Did

- Cleaned and explored raw sensor data from marine engines
- Created a **binary risk label** based on temperature thresholds
- Trained a **Random Forest Classifier**
- Evaluated the model with confusion matrix, accuracy, precision, and recall
- Visualized feature importances and classification results

---

## 📊 Model Performance

| Metric      | Value |
|-------------|-------|
| Accuracy    | 100%  |
| Precision   | 100%  |
| Recall      | 100%  |
| F1-score    | 100%  |

Confusion Matrix and Feature Importance plots are available in the `/outputs` folder.

---

## 🛠 Technologies Used

- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (RandomForestClassifier, metrics)
- Jupyter Notebook
- Visual Studio Code / Thonny

---

## 🚢 Why It Matters

Predictive maintenance in marine engines ensures:
- Reduced downtime
- Safer operations
- Cost-effective resource management
- Data-driven decision making

This project demonstrates how **simple classification models** can help prevent critical failures using real-time data.

---

## 📈 Future Improvements

- Incorporate time-based features (running period, time since last maintenance)
- Use ensemble models or neural networks for complex scenarios
- Build a dashboard for live monitoring of engine risk levels

---

## 👩‍💻 Author

**Bahar Işılar**  
Industrial Engineering Student
[LinkedIn](https://linkedin.com/in/bahar-isilar) • [GitHub](https://github.com/bhr34)

