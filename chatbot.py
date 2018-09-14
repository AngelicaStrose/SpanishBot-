import nltk 
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer() 

import numpy as np
import tflearn 
import tensorflow as tf
import random 
import json 


with open('intents.json') as json_data: 
  intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', '.']

for intent in intents['intents']:
  for pattern in intent['patterns']: 
    w = nltk.word_tokenize(pattern) 
    words.extend(w) 
    documents.append((w, intent['tag'])) 
    if intent['tag'] not in classes: 
       classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words))) 

classes = sorted(list(set(classes))) 

#print (len(documents), "documents") 
#print (len(classes), "classes", classes) 
#print (len(words), "unique stemmed words", words) 

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
x = list(training[:,0])
y = list(training[:,1])

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(x[0])]) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, len(y[0]), activation='softmax') 
net = tflearn.regression(net) 

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs') 
model.fit(x, y, n_epoch=1000, batch_size=8, show_metric=True) 
model.save('model.tflearn') 


def clean_up_sentence(sentence): 
  sentence_words = nltk.word_tokenize(sentence) 
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words 

def bow(sentence, words, show_details=False): 
  sentence_words = clean_up_sentence(sentence) 
  bag = [0] * len(words) 
  for s in sentence_words: 
    for i,w in enumerate(words): 
       if w==s: 
          bag[i] = 1
          if show_details: 
            print ("Found in bag: %s" % w) 
  return (np.array(bag)) 

context = {}

ERROR_THRESHOLD = 0.25 
def classify(sentence): 
  results = model.predict([bow(sentence, words)])[0]
  results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  results.sort(key=lambda x: x[1], reverse=True) 
  return_list = []
  for r in results:
     return_list.append((classes[r[0]], r[1])) 
  return return_list 

def response(sentence, userID='123', show_details=False): 
   results = classify(sentence) 
   if results: 
      while results: 
         for i in intents['intents']: 
            if i['tag'] == results[0][0]: 
               if 'context_set' in i: 
                   if show_details: print ('context:', i['context_set']) 
                   context[userID] = i['context_set'] 
               if not 'context_filter' in i or \
                  (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]): 
                   if show_details: print ('tag:', i['tag'])
                   return print(random.choice(i['responses'])) 
          results.pop(0) 

response ('How do you say hi in Spanish?') 
