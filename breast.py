import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('breast-cancer-wisconsin-data.csv')  # Replace with your actual file path
# If you're copying the data directly, you can use:
# data = pd.read_csv('your_data.csv', delimiter=',')

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Convert diagnosis to binary (M=1, B=0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Select relevant features based on common importance in breast cancer prediction
selected_features = [
    'radius_mean',        # Tumor size
    'texture_mean',       # Texture variation
    'perimeter_mean',     # Tumor perimeter
    'area_mean',          # Tumor area
    'smoothness_mean',    # Surface smoothness
    'compactness_mean',   # Tumor compactness
    'concavity_mean',     # Tumor concavity
    'concave points_mean' # Number of concave portions
]

# Prepare features (X) and target (y)
X = data[selected_features]
y = data['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': model.coef_[0]
})
feature_importance['Absolute_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Example of predicting a single sample
sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample)
probability = model.predict_proba(sample)
print(f"\nSample Prediction:")
print(f"Predicted class: {'Malignant' if prediction[0] == 1 else 'Benign'}")
print(f"Probability of Malignant: {probability[0][1]:.4f}")