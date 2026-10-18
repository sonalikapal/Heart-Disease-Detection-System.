# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv('heart.csv')

# Features and target
X = df.drop('target', axis=1)  
y = df['target']               

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

print("Model trained and saved as heart_disease_model.pkl")
