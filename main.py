from flask import Flask, request, jsonify, render_template
import re
from nltk.stem.porter import PorterStemmer
import pickle

# Stopword list for Indonesian
indonesian_stopwords = [
    'yang', 'untuk', 'dari', 'dengan', 'akan', 'pada', 'ini', 'itu', 'dan', 'di', 'ke', 'dalam', 'adalah', 
    'tidak', 'saya', 'dia', 'kita', 'kami', 'anda', 'mereka', 'apa', 'bagaimana', 'mengapa', 'siapa', 
    'bukan', 'ada', 'juga', 'sudah', 'belum', 'lebih', 'kurang', 'hanya', 'bisa', 'dapat', 'harus', 
    'banyak', 'sedikit', 'seperti', 'agar', 'tetapi', 'kalau', 'jadi', 'karena', 'oleh', 'menjadi', 
    'apakah', 'saat', 'setelah', 'sebelum', 'itu', 'tersebut', 'lagi', 'yg', ''
]
STOPWORDS = set(indonesian_stopwords)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    try:
        # Single string prediction
        text_input = request.form["text"]
        predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
        return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
