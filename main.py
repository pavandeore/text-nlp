from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    # Get the JSON data from the request
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "Text field not found in request"}), 400

    text = data['text']

    # Get the sentiment scores
    sentiment_scores = analyzer.polarity_scores(text)

    # Interpret the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    response_data = {
        "text": text,
        "sentiment": sentiment,
        "positive_score": sentiment_scores['pos'],
        "negative_score": sentiment_scores['neg'],
        "neutral_score": sentiment_scores['neu'],
        "compound_score": sentiment_scores['compound']
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
