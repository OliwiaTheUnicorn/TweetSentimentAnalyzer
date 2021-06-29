from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["neg"] != 0:
            return render_template('analyzer.html', message="Negative😞😞")
        else:
            return render_template('analyzer.html', message="Positive🙂🙂")
    else:
        return render_template('analyzer.html', message="Hello🙂🙂")


if __name__ == '__main__':
    app.run()
