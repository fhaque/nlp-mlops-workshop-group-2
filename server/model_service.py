# import mlflow.pyfunc
# import pandas as pd
# from model.data_util import config
# from model.eval import summarise_it
import json
import requests


class ModelService():
    def __init__(self):
        # self.model = mlflow.pyfunc.load_model('./model')
        # self.model = load_model()
        pass
    
    # def predict(self, data):
    #     article_summary, out_of_vocab =  summarise_it(data)
    #     # article_summary, out_of_vocab = predict_from_text(data, self.model)
    #     return article_summary

    def predict(self, data):
        res = requests.post(
            'http://ec2-18-224-45-242.us-east-2.compute.amazonaws.com:5000/translator//translate',
            data = json.dumps([{'src': data, 'id': 100}])
        )
        print(res)
        return res.json()[0][0]["tgt"]