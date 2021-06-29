from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentimentAnalyzer import getSentiment, train, clearData, updateNegativeTweet, updatePostiveTweet

import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        train()
        inp = request.form.get("inp")
        result = getSentiment(inp)
        return render_template('analyzer.html', message=result)
    else:
        return render_template('analyzer.html', message="Hello🙂🙂")


if __name__ == '__main__':
    app.run()
