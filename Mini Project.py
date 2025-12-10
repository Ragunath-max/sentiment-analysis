import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model():

    data = pd.read_excel(r"C:\Users\RAGUNATH R\Desktop\Sentiment Analysis\reviews.xlsx")

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Excel file must contain 'text' and 'label' columns")

    data["label"] = data["label"].str.lower()

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\n========= MODEL EVALUATION =========")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("===================================\n")

    return model, vectorizer


def predict_sentiment(model, vectorizer, text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


if __name__ == "__main__":

    print("Training sentiment analysis model... ")
    model, vectorizer = train_model()
    print("Training complete ")

    while True:
        user_text = input("\nType a review (or 'q' to quit): ")

        if user_text.lower() == "q":
            print("Bye 👋")
            break

        if not user_text.strip():
            print("Empty text not allowed")
            continue

        sentiment = predict_sentiment(model, vectorizer, user_text)
        print("Predicted Sentiment:", sentiment)
