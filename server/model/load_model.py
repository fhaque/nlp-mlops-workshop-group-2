from model.data_util import config
from model.eval import summarise_it
#from data_util.batcher import Example, Batcher


if __name__ == "__main__":
    text =  'That coronavirus death rate, which is lower than earlier estimates , takes into account potentially milder cases that often go undiagnosed -- but its still far higher than the 0.1% of people who are killed by the flu.When undetected infections'
    
    article_summary, out_of_vocab =  summarise_it(text)
    

    print('Article: ',text, '\n==> Summary: [',article_summary,']\nOut of vocabulary: ', out_of_vocab)    
