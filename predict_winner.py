import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv(r'C:\Users\manik\OneDrive\Desktop\liv VS BOC\liverpool_bournemouth_final_dataset.csv')

# Target column
target = df.columns[-1]
print(f"Using target column: '{target}'")

# Drop rows with missing target
df = df.dropna(subset=[target])

# Encode target if needed
if df[target].dtype == 'object':
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

# Features and target
X = df.drop(columns=[target])
y = df[target].astype(int)

# Encode categorical features if any
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Predict probabilities
proba = model.predict_proba(X_test)
class_labels = model.classes_

# Map labels to outcome names
label_map = {
    0: "DRAW",
    1: "LIVERPOOL WIN PROBABILITY",
    2: "BOURNEMOUTH WIN PROBABILITY"
}

# Print formatted probabilities for each test sample
for i in range(len(X_test)):
    print(f"\nTest Sample {i+1}:")
    for label in [1, 2, 0]:  # Print Liverpool, Bournemouth, Draw in this order
        if label in class_labels:
            idx = list(class_labels).index(label)
            print(f"{label_map[label]} = {proba[i][idx]*100:.2f}%")
