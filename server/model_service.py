# import mlflow.pyfunc
# import pandas as pd
from model.data_util import config
from model.eval import summarise_it


class ModelService():
    def __init__(self):
        # self.model = mlflow.pyfunc.load_model('./model')
        # self.model = load_model()
        pass
    
    def predict(self, data):
        article_summary, out_of_vocab =  summarise_it(text)
        # article_summary, out_of_vocab = predict_from_text(data, self.model)
        return article_summary