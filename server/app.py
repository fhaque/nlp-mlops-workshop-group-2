import json
import sys

from flask import Flask, request, jsonify, render_template
import pyshorteners

from twitter_service import TwitterService
from article_service import ArticleService
from model_service import ModelService


s = pyshorteners.Shortener()
twitter_service = TwitterService()
article_service = ArticleService()
model_service = ModelService()

# import mlflow.pyfunc
# import pandas as pd

# Name of the apps module package
app = Flask(__name__)

# Load in the model at app startup
# model = mlflow.pyfunc.load_model('./model')

# Load in our meta_data
# f = open("./model/code/meta_data.txt", "r")
# load_meta_data = json.loads(f.read())

# Meta data endpoint
# @app.route('/', methods=['GET'])
# def meta_data():

# 	return jsonify(load_meta_data)

# @app.route('/hello', methods=['GET'])
# def hello():
#     return 'Hello, World'

def create_tweet_content(article_url, article_summary):
    url = s.tinyurl.short(article_url)
    url_len = len(url) 
    summary_len = len(article_summary)
    summary = article_summary
    if url_len + summary_len > 280:
        summary = summary[0:279 - 1 - url_len]

    return "{} {}".format(summary, url)

@app.route('/health', methods=['GET'])
def health():
    return 'check'

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    print({'request': json.dumps(req) })
    article_url = req['data']  
    
    try:
        article = article_service.get_article(article_url)
        print(article.text)
        summary = model_service.predict(article.text)
        tweet_content = create_tweet_content(article_url, summary)
        print("Tweet Content: ", tweet_content)
        status = twitter_service.tweet(tweet_content)
        tweet_url = twitter_service.create_tweet_url(status)
        # tweet_html = twitter_service.get_tweet_embed_html(status)
        return jsonify({
            "tweet": status.full_text,
            "tweet_url": tweet_url,
            # "tweet_html": tweet_html,
            "tweet_id": status.id_str
        })
    except Exception as e:
        print(e)
        return jsonify({
            "error": str(e)
        })

# Prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
# 	req = request.get_json()
	
# 	# Log the request
# 	print({'request': req})

# 	# Format the request data in a DataFrame
# 	inf_df = pd.DataFrame(req['data'])

# 	# Get model prediction - convert from np to list
# 	pred = model.predict(inf_df).tolist()

# 	# Log the prediction
# 	print({'response': pred})

# 	# Return prediction as reponse
# 	return jsonify(pred)

app.run(host='0.0.0.0', port=5000, debug=True)