import pickle
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')

with open('best_model.pkl', 'rb') as model_file, open('best_tfidf.pkl', 'rb') as tfidf_file:
    model = pickle.load(model_file)
    tfidf = pickle.load(tfidf_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_message = request.form['message']
    transformed = tfidf.transform([email_message])
    result = model.predict(transformed)[0]
    print(f"Model raw output: {result}")

    # Confidence score
    try:
        proba = model.predict_proba(transformed)[0]
        spam_prob = round(float(max(proba)) * 100, 1)
        ham_prob = round(100 - spam_prob, 1)
    except:
        spam_prob = 100.0 if (result == 1 or result == '1') else 0.0
        ham_prob = round(100 - spam_prob, 1)

    # Keyword highlights
    spam_keywords = [
        'free', 'win', 'winner', 'prize', 'click', 'claim', 'offer',
        'limited', 'urgent', 'congratulations', 'call', 'cash', 'money',
        'credit', 'loan', 'guarantee', 'discount', 'deal', 'buy now',
        'act now', 'apply now', 'verify', 'account', 'password', 'bank',
        'paypal', 'invoice', 'update', 'confirm', 'login', 'expire'
    ]

    words = email_message.split()
    highlighted = []
    found_keywords = []
    for word in words:
        clean = word.lower().strip('.,!?')
        if clean in spam_keywords:
            highlighted.append(f'<mark class="spam-word">{word}</mark>')
            if clean not in found_keywords:
                found_keywords.append(clean)
        else:
            highlighted.append(word)
    highlighted_text = ' '.join(highlighted)

    if result == 1 or result == '1' or str(result).lower() == 'spam':
        prediction = "SPAM"
    else:
        prediction = "NOT SPAM"

    return render_template('template.html',
                           prediction=prediction,
                           email_message=email_message,
                           highlighted_text=highlighted_text,
                           found_keywords=found_keywords,
                           spam_prob=spam_prob,
                           ham_prob=ham_prob)

if __name__ == '__main__':
    app.run(debug=True)