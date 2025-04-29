import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot  as plt
import seaborn as sns

df = pd.read_csv("/Users/merveebrardemirel/Desktop/practice_datasets/Kalp Krizi Riski/heart.csv")

df.hist(figsize=(18,12), bins=20, edgecolor="black")
plt.title("Data Histogram")
#plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df , x='Age' , y='Cholesterol' , hue='HeartDisease')
plt.title('Age-Cholesterol Relation')
#plt.show()


plt.figure(figsize=(6,6))
sns.countplot(x='HeartDisease' , data=df)
plt.title('Heart Disease Infection Rates')
plt.show()
