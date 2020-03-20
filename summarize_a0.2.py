#from sys import argv
import re
import csv
import codecs
import numpy as np
import pandas as pd
import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import KeyedVectors

# # !cp "/content/drive/My Drive/Colab files/"* ./  — to copy data for work
##

def text_to_wordlist(text):    
    """Convert words to lower case, split them, and clean punctuated words using a hard coded dictionary."""
    text = text.lower().split()
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    return(text)

########### COMMAND STRING PARAMETERS ##################
# A couple of command string parameters to format output and CUDA mode
parser = argparse.ArgumentParser()
parser.add_argument("--color", "-c", help="Add ANSI color to output",action="store_true")
parser.add_argument("--nocuda", "-n", help="Disable CUDA support",action="store_true")
args = parser.parse_args() # search for arguments in command line

# Turn on colors in output for readibility
if args.color:
    GreenColor = "\033[1;32;40m"
    BlueColor =  "\033[1;36;40m"
    ResetColor = "\033[1;0;0m"
    RedColor = "\033[1;31;40m"
    YellowColor = "\033[1;33;40m"
    print(f"{GreenColor}/!\\ Colored output... {ResetColor}")
else:
    print("Try: -c, --color  for colored output.","\n")   
# disable CUDA support if asked
if args.nocuda: 
        cuda = False
        print(f"{RedColor}/!\\ cpu option selected: CUDA disabled... {ResetColor}")
else:
    cuda = True
    print(f"{RedColor}Use -cpu option to disable CUDA.{ResetColor}","\n")        

########### INITIAL SETUP ##################

# Define file names with articles and titles
train_article_txt_filename = './data/train.article_test.txt'
train_titles_txt_filename  = './data/train.title_test.txt'
embedding_file_name='./embeddings/glove.6B.50d.w2vformat.txt'

########### LOAD DATA ####################
#df_train = pd.read_csv(train_csv)
#df_train.head()
# Using readline() 
articles_file = open(train_article_txt_filename, 'r') 
titles_file = open(train_titles_txt_filename, 'r') 

# read in the training data and clean the text
articles_txt = [] 
#texts_2 = []
labels_txt = []

# read articles and titles
i_count = 0
while True: 
    i_count += 1
    line = articles_file.readline() 
    articles_txt.append(text_to_wordlist(line))
    if not line: break
#    print("Line{}: {}".format(count, line.strip()))   
articles_file.close() 
print( f"{BlueColor}Articles readed {ResetColor}{i_count}" )
i_count = 0
while True: 
    i_count += 1
    line = titles_file.readline() 
    labels_txt.append(text_to_wordlist(line))
    if not line: break
#    print("Line{}: {}".format(count, line.strip()))   
titles_file.close() 

print( f"{BlueColor}Titles readed {ResetColor}{i_count}" )
#with codecs.open( train_csv, encoding='utf-8') as f:
#    reader = csv.reader(f, delimiter=',')
#    header = next(reader)
#    for values in reader:
#        texts_1.append(text_to_wordlist(values[3]))
#        texts_2.append(text_to_wordlist(values[4]))
#        labels.append(int(values[5]))
print(f'{BlueColor}Found {ResetColor}{len(articles_txt)}{BlueColor} texts in {ResetColor}{train_titles_txt_filename}')
print(f'{BlueColor}We visualize the actual data: {ResetColor}')
print(f'{BlueColor}Article: {ResetColor} ', articles_txt[0])
print(f'{BlueColor}Title: {ResetColor}', labels_txt[0])

################ TOKENIZING DATA ####################

# tokenizing and indexing in the same step: converting sentence into words, then into integers
tokenizer = Tokenizer(num_words=200000)
tokenizer.fit_on_texts(articles_txt) # Assigning indices to the words 
sequences_1 = tokenizer.texts_to_sequences(articles_txt) # Converting each article to sequences of indices
labels_x = tokenizer.texts_to_sequences(labels_txt) # Converting each title to sequences of indices
print(f"{BlueColor}(Articles) sequences_1: {ResetColor}{sequences_1[0]}")
print(f"{BlueColor}labels: {ResetColor}{labels_x[0]}")

word_index = tokenizer.word_index
print(f'{BlueColor}Found {YellowColor}{len(word_index)}{BlueColor} unique tokens{ResetColor}') # Vocabulary size
#print(f'{word_index}') # uncomment to see the word dictionary
#exit()
# padding/truncating sentences to the same length of 30 words
max_sequence_length = 40

# Create tensors of sequences and send them to GPU
# data1: data corresponding to articles 
# labels data corresponding to trtles of these articles    
if cuda: 
    data = torch.tensor(pad_sequences(sequences_1, maxlen=max_sequence_length)).cuda()
else:
    data = torch.tensor(pad_sequences(sequences_1, maxlen=max_sequence_length))  
if cuda: 
    labels = torch.tensor(pad_sequences(labels_x, maxlen=max_sequence_length)).cuda()
else:
    labels = torch.tensor(pad_sequences(labels_x, maxlen=max_sequence_length))         
# Test if conversion of words to indices the same in articles and titles:
wordtest = 'account' # Any word that is available in both texts
print(f"{BlueColor}Check conversion to indeces works: {ResetColor}")
print(f"{BlueColor}data: {ResetColor}{data[0]}")
print(f"{BlueColor}labels: {ResetColor}{labels[0]}")
print(f"{BlueColor}Test: The word {ResetColor}\'{wordtest}\'{BlueColor} index is {ResetColor}{torch.tensor(word_index[wordtest])}")

# Wonderful, lets load embedding model (already prepared)
print(f'{BlueColor} Loading {ResetColor}{embedding_file_name}{BlueColor} with embedding vectors... {ResetColor}')

embeddings_model = KeyedVectors.load_word2vec_format(embedding_file_name) #Load tensors for words from file
print(f'{BlueColor} Creating embedding mechanism: {ResetColor}')
glove_tensors = torch.FloatTensor(embeddings_model.vectors) # make them actual tensors
embedding = nn.Embedding.from_pretrained(glove_tensors) # create embedding mechanism
print(f"{BlueColor}Embedding: {ResetColor}",embedding) # print how many vectors loaded

# query the embeddings
# TODO: finish the below so that we can query the pre-trained embeddings. Utilize the word_index dictionary we created.
input = torch.tensor(word_index[ wordtest ]) # NOTE: assignment section (1)
# other example words: 'them', 'company', 'during'
print(f"{BlueColor}Word:{ResetColor}", wordtest ,end='')
print(f"{BlueColor} query the embedding: {ResetColor}\n{embedding(input)}")

# NOTE: if we feed question 1 and question 2 separately into 2 different networks, performance would improve
# But we take a simpler approach here by unifying the 2 questions into 1 sentence, and feed it into 1 network
#data = torch.cat((data_1, data_2), 1)
#data=torch.cat(data_1, 1)
#labels = torch.cat(labels, 1)
#data = data_1.clone().detach()
#torch.tensor(data_1)
#data = torch.tensor(data_1, data_2)
#labels = torch.tensor(labels_x)
print(f"{BlueColor}Data sizes check:{ResetColor}")
print(f"{BlueColor}data size: {ResetColor} {data.size()}")
#print(f"{BlueColor}labels_x size: {ResetColor} {labels_x.size()}")
print(f"{BlueColor}labels size: {ResetColor} {labels.size()}")

# split data into trainining, validation, testing set
split_frac = 0.8 # keep 80% of the data for training
len_feat = len(data) # length of the whole dataset

train_x = data[0:int(split_frac*len_feat)] # keep sentence 0 to (0.8 * length of the data) for training
train_y = labels[0:int(split_frac*len_feat)] # keep label 0 to (0.8 * length of the data) for training

temp_remain_x = data[int(split_frac*len_feat):] 
temp_remain_y = labels[int(split_frac*len_feat):]

# Use 10% of data for validation
valid_x = temp_remain_x[0:int(len(temp_remain_x)*0.5)]
valid_y = temp_remain_y[0:int(len(temp_remain_y)*0.5)]

# USe the other 10% of data for test
test_x = temp_remain_x[int(len(temp_remain_x)*0.5):]
test_y = temp_remain_y[int(len(temp_remain_y)*0.5):]

print(len(train_x), len(valid_x), len(test_x))
assert len(train_x) + len(valid_x) + len(test_x) == len(data), 'Two lengths not equal'

# Create and set up torch dataset
train_data = TensorDataset(train_x, train_y)
# TODO: complete the tensor setup for validation data
valid_data = TensorDataset(valid_x, valid_y) # NOTE: assignment section (2)
test_data = TensorDataset(test_x, test_y)
print(train_data, valid_data, test_data)

#print("Done.")
#exit()

# batching
batch_size = 2000
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# TODO: complete the batching for validation data
valid_loader = DataLoader(valid_data, shuffle=True,batch_size=batch_size) # NOTE: assignment section (3)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# sanity check to see if we have consisntet label of 0 and 1 for our train/val/test set
#print('labels: ', pd.Series(labels.numpy()).value_counts() / len(labels), '')
#print(type(train_y))
#print('train: ', pd.Series(train_y.numpy()).value_counts() / len(train_y), '')
#print('val: ', pd.Series(valid_y.numpy()).value_counts() / len(valid_y), '')
#print('test: ', pd.Series(test_y.numpy()).value_counts() / len(test_y), '')

# Here we do 1 more sanity check to see if we can get one batch of training data
iter_data = iter(train_loader)
temp_x, temp_y = iter_data.next()
print(f'{BlueColor}Here we do 1 more sanity check to see if we can get one batch of training data:{ResetColor}')
print('One batch of input: \n', temp_x)
print('One batch of input size: ', temp_x.size())
print('\n')
print('One batch of label: \n', temp_y)
print('One batch of label size: ', temp_y.size())

#print("Done.")
#exit()

###### LSTM

class GRU(nn.Module):
    """
    The RNN model that will be used to perform duplicate detection.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # option (1): trained together with LSTM
        # If you dont want to train the embedding layer, comment out the above line and 
        # un-comment the following line 
        # self.embedding = embedding # option (2): pre-trained
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # ===== Encoder  =====
        # embeddings and lstm_out
        enc_embeds = self.embedding(x)
        
        # TODO: complete the forward propagation for LSTM
        gru_out, hidden = self.gru(enc_embeds)# NOTE: assignment section (4)
    
        # stack up lstm outputs
        # creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        gru_out = gru_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        # TODO: complete the drop out layer
        out = self.dropout(gru_out) # NOTE: assignment section (5)
        out = self.fc(out)
        
        # ===== Decoder =====
        #dec_embeds = 
        #lstm_out, hidden = self.lstm(enc_embeds)

        gru_out, hidden = self.gru(hidden)# hidden state from GRU to decoder
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        # convert 2D to 1D
        sig_out = sig_out[:, -1]
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data # grabbing the entire model parameter set
        if cuda:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

vocab_size = len(word_index) + 1 # +1 for the 0 padding
output_size = 40
embedding_dim = 50
hidden_dim = 256
n_layers = 2
net = GRU(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

#print("Done.")
#exit()

# training parameters
lr = 0.001
# TODO: specify binary cross entropy loss
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss() # NOTE: assignment section (6)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
epochs = 10
counter = 0
clip = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

#print("Done.")
#exit()


for e in range(epochs):
    net.train()
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        if len(inputs) != batch_size:
          continue
        counter += 1
        if counter % 100 == 99:
            print(".",end='') 
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        inputs = inputs.to(device)
        labels = labels.to(device)
        output, h = net(inputs, h)
        labels = labels[:, -1]
        # calculate the loss and perform backprop
        # TODO: calculate the loss between predicted and labels
        print(f"output: {YellowColor} {output.float().size()} {ResetColor}")
        print(f"labels: {YellowColor} {labels.squeeze().float().size()} {ResetColor}")
        print(f"output: {YellowColor} {output.float()[0]} {ResetColor}")
        print(f"labels: {YellowColor} {labels.squeeze().float()[0]} {ResetColor}")

        loss = criterion(output.float(), labels.squeeze().float()) # NOTE: assignment section (7)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

    # Get validation loss
    val_h = net.init_hidden(batch_size)
    val_losses = []
    val_acc = []
    net.eval()
    for inputs, labels in valid_loader:
        if len(inputs) != batch_size:
          continue

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in val_h])

        inputs = inputs.type(torch.LongTensor)
#        inputs, labels = inputs, labels
        inputs, labels = inputs.cuda(), labels.cuda()
        output, val_h = net(inputs, val_h)
        val_loss = criterion(output.squeeze(), labels.float())
        
        # accuracy
        output = (output > 0.5).float()
        output = output.type(torch.LongTensor)
        correct = (output.cpu() == labels.cpu()).float().sum()

        val_losses.append(val_loss.item())
        val_acc.append(correct / output.shape[0])

    print("Epoch: {}/{}...".format(e+1, epochs),
          "Step: {}...".format(counter),
          "Loss: {:.6f}...".format(loss.item()),
          "Val Loss: {:.6f}".format(np.mean(val_losses)),
          "Val Accuracy: {:.6f}".format(np.mean(val_acc)))

print("Done.")

# check accuracy
test_h = net.init_hidden(batch_size)
test_losses = []
test_acc = []
net.eval()
for inputs, labels in test_loader:
    if len(inputs) != batch_size:
      continue

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    test_h = tuple([each.data for each in test_h])

    inputs = inputs.type(torch.LongTensor)
    inputs, labels = inputs.cuda(), labels.cuda()
    output, test_h = net(inputs, test_h)
    test_loss = criterion(output.squeeze(), labels.float())
    output = (output > 0.5).float()
    output = output.type(torch.LongTensor)
    correct = (output.cpu() == labels.cpu()).float().sum()

    test_losses.append(test_loss.item())
    test_acc.append(correct / output.shape[0])

net.train()
print("Test Loss: {:.6f}".format(np.mean(test_losses)),
      "\n"
      "Test Accuracy: {:.6f}".format(np.mean(test_acc))
      )

PATH = 'model/lstm.pt'
torch.save(net.state_dict(), PATH)

