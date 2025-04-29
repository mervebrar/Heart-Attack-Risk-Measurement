import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

df = pd.read_csv('/Users/merveebrardemirel/Desktop/practice_datasets/Kalp Krizi Riski/heart.csv')


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])


X = df.drop('HeartDisease' , axis=1)
y = df['HeartDisease']


X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.3 , random_state=42)


smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

print('Model 1: Artificial Neural Networks')
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1] ,activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy' , optimizer = optimizer , metrics=['accuracy'])
model.fit(X_train_scaled, y_train_smote, epochs=100 , verbose=1 , validation_data=(X_test_scaled, y_test) )
loss, accuracy = model.evaluate(X_test_scaled, y_test)

y_pred_ann_prob = model.predict(X_test_scaled)
y_pred_ann = (y_pred_ann_prob > 0.5).astype(int)


print(classification_report(y_test, y_pred_ann))
