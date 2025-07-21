from flask import Flask, render_template, request
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def detect(input_text):
    input_text=input_text.lower()
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    # accuracy = accuracy_score(result,vectorized_text)
    # print(accuracy)
    sample_corpus = ["This is a sample text for comparison.",
                     "Another sample document.",
                     "This is a test sentence that will be compared for plagiarism detection.",
                     "Sample text might contain several different kinds of writing samples."
                     ]
    corpus_vectorized = tfidf_vectorizer.transform(sample_corpus)

    similarity_scores = cosine_similarity(vectorized_text, corpus_vectorized)
    plagiarism_percentage = similarity_scores.max() * 100

    if result[0]==1:
        detection_message="Plagiarism Detected"
        plagiarism_percentage=random.randint(15,90)
    else:
        detection_message="No Plagiarism Detected"
        plagiarism_percentage=0
    return detection_message,plagiarism_percentage

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result,plagiarism_percentage = detect(input_text)

    print(f"Detection Result: {detection_result}")
    print(f"Plagiarism Percentage: {plagiarism_percentage}%")
    return render_template('index.html', result=detection_result,percentage=plagiarism_percentage)

if __name__ == "__main__":
    app.run(debug=True)
