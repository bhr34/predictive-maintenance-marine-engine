import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "C:/Users/Asus/Downloads/marine_engine_data.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.columns)

# Örnek risk kuralı: sıcaklık 85 üzeriyse riskli
df['Risk'] = df['engine_temp'].apply(lambda x: 'risk' if x > 95 else 'normal')
df['Risk'].value_counts()

df.to_csv("C:/Users/Asus/Downloads/marine_engine_data_labeled.csv", index=False)
print("Risk column added. New dataset saved.")

# Kutu grafiği
sns.boxplot(data=df, x='Risk', y='engine_temp', color='steelblue')
plt.title("Engine Temperature by Risk Status")
plt.xlabel("Risk")
plt.ylabel("Engine Temperature (°C)")
plt.show()
