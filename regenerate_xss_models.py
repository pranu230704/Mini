import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load your XSS dataset
# Replace with your actual dataset path
df = pd.read_csv('path_to_your_xss_dataset.csv')

# Assuming columns: 'payload' and 'label' (0=Normal, 1=XSS)
X = df['payload']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Save models with protocol 4 (more compatible)
with open('models/xss_random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f, protocol=4)

with open('models/xss_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=4)

print("✔ XSS models regenerated successfully!")