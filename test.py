import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# New email
email = ["Meeting tomorrow at 10 AM"]

# Convert to numbers
email_vec = vectorizer.transform(email)

# Predict
result = model.predict(email_vec)

print("Prediction:", result[0])