import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'text': [
        'I love this product!',
        'This is the worst experience ever.',
        'Absolutely fantastic service.',
        'I hate this so much.',
        'Not bad, could be better.',
        'I am extremely happy with this.',
        'Terrible, I will never use this again.',
        'This is okay, nothing special.',
        'Amazing quality and great support.',
        'Awful, very disappointed.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'neutral', 'positive', 'negative', 'neutral', 'positive', 'negative']
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Preprocessing
X = df['text']
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Make predictions
y_pred = model.predict(X_test_vectors)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with new input
new_text = ["I really enjoy using this!", "This is the worst thing I've bought."]
new_text_vectors = vectorizer.transform(new_text)
predictions = model.predict(new_text_vectors)
print("\nPredictions for new text:", predictions)