import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load Dataset
df_fake = pd.read_csv('data/Fake.csv')
df_true = pd.read_csv('data/True.csv')

# 2. Create labels
df_fake['label'] = 'fake'
df_true['label'] = 'true'

# 3. Merge datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# 4. Separate features and labels
X = df['text']
y = df['label']

# 5. Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) # ignores words that appear in more than 70% of news stories
X_train_tfidf = vectorizer.fit_transform(X_train)

# 7. Train model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 8. Save model and vectorizer
joblib.dump(model, 'modelo_fake_news.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model saved!")
