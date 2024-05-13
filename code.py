import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier  # Import XGBoost classifier

# Load the dataset
data = pd.read_csv(
    "Downloads/sentiment140.csv",
    names=["target", "ids", "date", "flag", "user", "text"],
    encoding="latin-1",
)

# Preprocess the data
data = data[["text", "target"]]
data["target"] = data["target"].apply(lambda x: int(x))
data = data[data["target"] != 2]  # Remove neutral tweets
data["target"] = data["target"].apply(lambda x: 1 if x == 4 else 0)  # Convert to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["target"], test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_vectorized, y_train)

# Train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train_vectorized, y_train)

# Predict probabilities for all models
nb_probs = nb_model.predict_proba(X_test_vectorized)
logistic_probs = logistic_model.predict_proba(X_test_vectorized)
xgb_probs = xgb_model.predict_proba(X_test_vectorized)

# Combine probabilities using a simple average
combined_probs = (nb_probs + logistic_probs + xgb_probs) / 3

# Predict classes based on combined probabilities
y_pred = (combined_probs[:, 1] > 0.5).astype(int)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

import re
import string


def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    return tweet


# Accept input tweet from the user
input_tweet = input("Enter the tweet to analyze: ")

# Preprocess the input tweet
preprocessed_tweet = preprocess_tweet(input_tweet)

# Transform the preprocessed tweet into numerical features
input_tweet_vectorized = vectorizer.transform([preprocessed_tweet])

# Use the trained ensemble model to predict the sentiment of the input tweet
nb_prob = nb_model.predict_proba(input_tweet_vectorized)
logistic_prob = logistic_model.predict_proba(input_tweet_vectorized)
xgb_prob = xgb_model.predict_proba(input_tweet_vectorized)

# Combine probabilities using a simple average
combined_prob = (nb_prob + logistic_prob + xgb_prob) / 3

# Predict class based on combined probabilities
predicted_sentiment = "positive" if combined_prob[:, 1] > 0.5 else "negative"

# Print the predicted sentiment
print(f"The predicted sentiment of the tweet is: {predicted_sentiment}")
