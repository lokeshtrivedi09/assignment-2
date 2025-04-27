
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load dataset
data = pd.read_csv('Crop_recommendation.csv')

# Features and target
X = data.drop('label', axis=1)
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
with open('model/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
