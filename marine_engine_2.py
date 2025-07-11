from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/Users/Asus/Downloads/marine_engine_data_labeled.csv")

# Hedef değişken ve bağımsız değişkenler
X = df[['engine_temp', 'vibration_level', 'rpm', 'fuel_consumption', 'oil_pressure']]
y = df['Risk']

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['risk', 'normal'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['risk', 'normal'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
