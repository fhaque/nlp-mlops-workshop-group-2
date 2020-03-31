import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import torch as T
#import mlflow 
#import mlflow.pytorch
import torch.nn as nn
import torch.nn.functional as F
from model.model import Model

from model.data_util import config, data
from model.data_util.batcher import Batcher
from model.data_util.data import Vocab
from model.train_util import *
from model.beam_search import *
# from rouge import Rouge
import argparse

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        # time.sleep(5)
        self.setup_valid()

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        if T.cuda.is_available(): map_location = T.device('cuda')
        else:  map_location = T.device('cpu')
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model),map_location)
        self.model.load_state_dict(checkpoint["model_dict"])
#        mlflow.pytorch.save_model(self.model,config.save_model_path+'_2')
#        mlflow.pytorch.load_model(config.save_model_path+'_2')

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = "test_"+loadfile.split(".")[0]+".txt"
    
        with open(os.path.join("data",filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: "+article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, print_sents = False):

        # self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        # rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)

            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
                article_art_oovs = batch.art_oovs[i]

            #batch = self.batcher.next_batch()
            break

        load_file = self.opt.load_model # just a model name

        #if print_sents:
        #    self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)
        Batcher.article_summary = decoded_sents[0]
        Batcher.oovs = " ".join(article_art_oovs)

#        print('Article: ',article_sents[0], '\n==> Summary: [',decoded_sents[0],']\nOut of vocabulary: ', " ".join(article_art_oovs),'\nModel used: ', load_file)    
        scores = 0 #rouge.get_scores(decoded_sents, ref_sents, avg = True)
        if self.opt.task == "test": 
            print('Done.')
            #print(load_file, "scores:", scores)
        else:
            print('nothing')
            # rouge_l = scores["rouge-l"]["f"]
            #print(load_file, "rouge_l:", "%.4f" % rouge_l)

class opt(object):
    def __init__(self):
        self.task =""
        self.loadmodel = ""  

def summarise_it(input_string="Short text"):
    Batcher.initial_article = input_string
    opt.task = "test"
    opt.load_model = "0003699.tar"
    Evaluate(config.test_data_path, opt).evaluate_batch()
    return Batcher.article_summary, Batcher.oovs

# def load_model():
#     Batcher.initial_article = "dummy"
#     opt.task = "test"
#     opt.load_model = "0003699.tar"
#     evaluator = Evaluate(config.test_data_path, opt)
#     # evaluator.evaluate_batch()
#     return evaluator

# def predict_from_text(text, evaluator):
#     Batcher.initial_article = text
#     # evaluator.evaluate_batch()
#     return Batcher.article_summary, Batcher.oovs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate","test"])
    parser.add_argument("--start_from", type=str, default="0020000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    opt = parser.parse_args()
    print(type(opt))
    if opt.task == "validate":
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt)
            eval_processor.evaluate_batch()
    else:   #test
        eval_processor = Evaluate(config.test_data_path, opt)
        eval_processor.evaluate_batch()
