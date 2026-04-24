import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("email_dataset.csv")

# select columns
X = data["text"]
y = data["label"]

# clean text
X = X.astype(str)
X = X.str.replace("Subject:", "", regex=False)

# convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# test
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved ✔")