from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentimentAnalyzer import getSentiment, train, clearData, updateNegativeTweet, updatePostiveTweet, initializeFiles

import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        if (request.values.get("action") == 'Reset'):
            initializeFiles()
            result = "Sentiment Analyzer is initialized. Please train the model"
            inputValue = ""
        elif (request.values.get("action") == 'Train'):
            train()
            result = "Sentiment Analyzer training is complete"
            inputValue = ""
        elif (request.values.get("action") == 'Clear'):
            result = "Welcome to Tweet Sentiment Analyzer🙂🙂"
            inputValue = ""
        elif (request.values.get("action") == 'Submit'):
            inputValue = request.form.get("inp")
            if not inputValue:
                result = "Input is empty"
            else:
                result = getSentiment(inputValue)
        elif (request.values.get("action") == 'Feedback'):
            inputValue = request.form.get("inp")
            if not inputValue:
                result = "Input is empty"
            else:
                if (request.form.get('decision') == 'Positive'):
                    updatePostiveTweet(inputValue)
                else:
                    updateNegativeTweet(inputValue)
                result = "Feedback has been added. Please train the model again"

        return render_template('analyzer.html', message=result, input=inputValue)
    else:
        return render_template('analyzer.html', message="Welcome to Tweet Sentiment Analyzer🙂🙂", input="")


if __name__ == '__main__':
    app.run()
