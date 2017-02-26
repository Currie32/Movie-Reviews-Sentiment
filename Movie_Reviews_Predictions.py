
# coding: utf-8

# # Predicting the Sentiment of Movie Reviews

# The goal for this analysis is to predict if a review rates the movie positively or negatively. Inside this dataset there are 25,000 labeled movies reviews for training, 50,000 unlabeled reviews for training, and 25,000 reviews for testing. More information about the data can be found at: https://www.kaggle.com/c/word2vec-nlp-tutorial.
# 
# This data comes from the 2015 Kaggle competition, "Bag of Words Meets Bags of Popcorn." Despite the competition being finished, I thought it could still serve as a useful tool for my first Natural Lanugage Processing (NLP) project. Within this analysis you will find three methods for predicting the sentiment of movie reviews. I wanted to experiment with a few strategies to gain an understanding of different strategies and compare their results. The three methods that I will use are: 
# - Bag of Centroids with Random Forest
# - Bag of Words with Tensorflow
# - Long Short Term Memory (LSTM) with Tensorflow
# 

# In[1]:

import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk.data
import logging  
from gensim.models import Word2Vec
import multiprocessing
import time
import tflearn
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from bs4 import BeautifulSoup


# ## Load and Explore the Data

# In[2]:

# Load the Data
train = pd.read_csv("labeledTrainData.tsv", 
                    header=0, 
                    delimiter="\t", 
                    quoting=3 )

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", 
                              header=0, 
                              delimiter="\t", 
                              quoting=3 )

test = pd.read_csv("testData.tsv", 
                   header=0, 
                   delimiter="\t", 
                   quoting=3 )


# In[3]:

# Compare the lengths of the datasets
print(train.shape)
print(unlabeled_train.shape)
print(test.shape)


# In[4]:

train.head()


# In[5]:

# Take a look at a review
train.review[0]


# In[6]:

unlabeled_train.head()


# In[7]:

unlabeled_train.review[0]


# In[8]:

test.head()


# In[9]:

test.review[0]


# Everything looks good with the data. Naturally the reviews are of different lengths, but everything is as expected.

# ## Model #1: Bag of Centroids

# In[10]:

def review_to_wordlist(review, remove_stopwords=False):
    # Clean the text, with the option to remove stopwords.

    # Remove HTML
    review_text = BeautifulSoup(review).get_text()

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " \'s", review_text)
    review_text = re.sub(r"\'ve", " \'ve", review_text)
    review_text = re.sub(r"n\'t", " n\'t", review_text)
    review_text = re.sub(r"\'re", " \'re", review_text)
    review_text = re.sub(r"\'d", " \'d", review_text)
    review_text = re.sub(r"\'ll", " \'ll", review_text)
    review_text = re.sub(r",", " , ", review_text)
    review_text = re.sub(r"!", " ! ", review_text)
    review_text = re.sub(r"\(", " \( ", review_text)
    review_text = re.sub(r"\)", " \) ", review_text)
    review_text = re.sub(r"\?", " \? ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)

    # Convert words to lower case and split them
    words = review_text.lower().split()

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    # Shorten words to their stems (i.e. remove suffixes and other word endings)
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # Return a list of words
    return(stemmed_words)


# In[11]:

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Split a review into parsed sentences
    # Returns a list of sentences, where each sentence is a list of words
    
    # Use the NLTK tokenizer to split the review into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    
    # Return the list of sentences
    # Each sentence is a list of words, so this returns a list of lists
    return sentences


# In[12]:

sentences = [] 

print ("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# In[13]:

# Check how many sentences we have in total 
print (len(sentences))
print()
print (sentences[0])
print()
print (sentences[1])


# In[14]:

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 250      # Word vector dimensionality                      
min_word_count = 20     # Minimum word count                        
num_workers = 1         # Number of threads to run in parallel
context = 20            # Context window size                                                                                    
downsampling = 1e-3     # Downsample setting for frequent words

# Initialize and train the model
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, 
                          workers = num_workers,
                          size = num_features,
                          min_count = min_word_count,
                          window = context, 
                          sample = downsampling)

# Call init_sims because we won't train the model any further 
# This will make the model much more memory-efficient.
model.init_sims(replace=True)

# save the model for potential, future use.
model_name = "250features_20minwords_20context"
model.save(model_name)


# In[15]:

# Load the model, if necessary
# model = Word2Vec.load("250features_20minwords_20context") 


# In[16]:

# Take a look at the performance of the model
print(model.most_similar("man"))


# In[17]:

model.most_similar("great")


# In[18]:

model.most_similar("terribl")


# In[19]:

model.most_similar("movi")


# In[20]:

model.most_similar("best")


# The model looks good so far. Each of these words has appropriate similar words.

# In[21]:

model.syn0.shape


# In[22]:

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters,
                           verbose = 2)
idx = kmeans_clustering.fit_predict(word_vectors)


# In[23]:

# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number                                                                                            
word_centroid_map = dict(zip(model.wv.index2word, idx))


# In[24]:

def create_bag_of_centroids(wordlist, word_centroid_map):
    
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    
    return bag_of_centroids


# In[25]:

# Clean the training and testing reviews, remove stopwords.
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
print("Training reviews are clean")  

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
print("Testing reviews are clean") 


# In[26]:

clean_train_reviews[0]


# In[27]:

clean_test_reviews[0]


# In[28]:

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

print("Training reviews are complete.")    
    
# Repeat for test reviews 
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map )
    counter += 1
    
print("Testing reviews are complete.")  


# In[29]:

# Split the data for testing
x_train, x_test, y_train, y_test = train_test_split(train_centroids,
                                                    train.sentiment,
                                                    test_size = 0.2,
                                                    random_state = 2)


# In[32]:

# Use GridSearchCV to find the optimal parameters
parameters = {'n_estimators':[100, 200, 300],
              'max_depth':[1,3,5,7, None],
              'min_samples_leaf': [1,3,5],
              'verbose': [True]}

# Use Random Forest to make the predictions
forest = RandomForestClassifier()
grid = GridSearchCV(forest, parameters)
grid.fit(x_train, y_train)


# In[33]:

print("Best training score = ", grid.best_score_)

grid_predictions = grid.best_estimator_.predict(x_test)
grid_score = metrics.accuracy_score(y_test, grid_predictions) 

print("Accuracy: {0:f}".format(grid_score))

print("Best Parameters = ", grid.best_params_)


# We have a pretty good first result here. The original code supplied by Google in the tutorial scored about 84%. It's nice that we have made some improvements to it, and scored higher.

# In[34]:

forest = RandomForestClassifier(n_estimators = 300,
                                max_depth = None,
                                min_samples_leaf = 1, 
                                verbose = 2)

# Apply the Random Forest Model to the full training data.
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)


# When I submit the results to the Kaggle competition its accuracy is 85.7%, which ranks it near the middle of the pack.

# ## Model #2: Bag of Words

# In[35]:

# Find the length of each training and testing review
review_lengths = []
for review in clean_train_reviews:
    review_lengths.append(len(review))
    
review_lengths_test = []
for review in clean_test_reviews:
    review_lengths_test.append(len(review))


# In[36]:

# Change the lists to dataframes so that describe() can be used
review_lengths = pd.DataFrame(review_lengths)
review_lengths_test = pd.DataFrame(review_lengths_test)


# In[37]:

# Print out a summary of the review lengths
print("Summary Training Reviews:")
print(review_lengths.describe())
print()
print("Summary Testing Reviews:")
print(review_lengths_test.describe())


# In[38]:

# Find the maximum number of words for a percentile
percentile = 90
print(np.percentile(review_lengths, percentile))
print(np.percentile(review_lengths_test, percentile))


# In[39]:

# Join the list of words to make more natural sentences
clean_train_reviews_sentences = []
space = " "
for review in clean_train_reviews:
    sentence = space.join(review)
    clean_train_reviews_sentences.append(sentence)
    
clean_test_reviews_sentences = []
space = " "
for review in clean_test_reviews:
    sentence = space.join(review)
    clean_test_reviews_sentences.append(sentence)


# In[40]:

# Take a look at a review to ensure everything is alright
clean_train_reviews_sentences[0]


# In[41]:

# Split the data in training and testing
x_train, x_test, y_train, y_test = train_test_split(clean_train_reviews_sentences, 
                                                    train.sentiment, 
                                                    test_size = 0.2, 
                                                    random_state = 2)


# In[42]:

# Process the reviews to limit the number of words and length

# max_document_length = maximum number of words in a review
# min_frequency = minimum number of times a word must be present to be used in the vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = 281,
                                                          min_frequency = 5)
x_train_transformed = np.array(list(vocab_processor.fit_transform(x_train)))
x_test_transformed = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


# In[43]:

# Check to make sure everything looks okay
x_train_transformed[0]


# In[44]:

print(x_train[0])
print()
print(len(x_train[0].split()))


# In[45]:

EMBEDDING_SIZE = 15

def bag_of_words_model(features, target):    
    # One-hot encode the target feature - positive and negative
    target = tf.one_hot(target, 2, 1, 0)  
    
    # If you alter the original n_words, you will need to input the value manually.
    features = tf.contrib.layers.bow_encoder(features, 
                                             vocab_size = 18307, #n_words, 
                                             embed_dim = EMBEDDING_SIZE)  
    
    logits = tf.contrib.layers.fully_connected(features, 
                                               2, 
                                               activation_fn=None)  
    
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    
    train_op = tf.contrib.layers.optimize_loss(loss, 
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adam', 
                                               learning_rate=0.005)  
    
    return ({'class': tf.argmax(logits, 1), 
             'prob': tf.nn.softmax(logits)},      
            loss, train_op)


# In[46]:

# Set classifier as bag_of_words_model
classifier_bow = learn.Estimator(model_fn = bag_of_words_model) 
# Train model
classifier_bow.fit(x_train_transformed, y_train, steps=1000) 


# In[47]:

bow_predictions = [p['class'] for p in classifier_bow.predict(x_test_transformed, as_iterable=True)] 
score = metrics.accuracy_score(y_test, bow_predictions) 
print("Accuracy: {0:f}".format(score))


# The prediction of 85.06% is pretty similar to the Bag of Centroids model. Let's see how it compares when we use all of the data.

# In[48]:

# Process the full data set to make final predictions
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = 281,
                                                          min_frequency = 5)
x_train_all = np.array(list(vocab_processor.fit_transform(clean_train_reviews_sentences)))
x_test_all = np.array(list(vocab_processor.transform(clean_test_reviews_sentences)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


# In[49]:

# Need to use learn.Estimator again to 'reset' the model. 
# Otherwise you would be 'double training.'
classifier_bow = learn.Estimator(model_fn = bag_of_words_model) 
classifier_bow.fit(x_train_all, train.sentiment, steps=1000) 

result_bow = [p['class'] for p in classifier_bow.predict(x_test_all, as_iterable=True)] 

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result_bow})
output.to_csv("BagOfWords.csv", index=False, quoting=3)


# The accuracy drops when we use the Kaggle predictions to 83.7%. 

# ## Model #3: LSTM

# In[60]:

EMBEDDING_SIZE = 25
LTSM_SIZE = 25
number_of_layers = 3

def rnn_model(features, target):  
    """RNN model to predict from sequence of words to a class."""  
    
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE]
    word_vectors = tf.contrib.layers.embed_sequence(features, 
                                                    vocab_size = 9600, #n_words, 
                                                    embed_dim = EMBEDDING_SIZE)   
    
    # Split into list of embeddings per word, while removing doc length dim.
    word_list = tf.unstack(word_vectors, axis=1)
    
    # Create a Long Short Term Memory cell with hidden size of LISTM_SIZE.
    cell = tf.nn.rnn_cell.BasicLSTMCell(LTSM_SIZE, state_is_tuple=False)
    
    # Create an unrolled Recurrent Neural Networks to length of
    # max_document_length and passes word_list as inputs for each unit.
    _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)   

    target = tf.one_hot(target, 2, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 2, activation_fn=None)  
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)   
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(loss, 
                                               tf.contrib.framework.get_global_step(),      
                                               optimizer='Adam', 
                                               learning_rate=0.005, 
                                               clip_gradients=1.0)   
    return ({'class': tf.argmax(logits, 1), 
             'prob': tf.nn.softmax(logits)},      
             loss, train_op)


# In[51]:

# Need to process the reviews again, because my laptop will crash if the network is too large.
# If you are using a GPU or have more than 8GB of RAM, you should be able to use more data for training.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = 150,
                                                          min_frequency = 20)
x_train_transformed = np.array(list(vocab_processor.fit_transform(x_train)))
x_test_transformed = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


# In[57]:

classifier_rnn = learn.Estimator(model_fn = rnn_model) 
classifier_rnn.fit(x_train_transformed, y_train, steps = 500) 

predictions_rnn = [p['class'] for p in classifier_rnn.predict(x_test_transformed, as_iterable=True)] 
score = metrics.accuracy_score(y_test, predictions_rnn) 
print("Accuracy: {0:f}".format(score))


# The testing accuracy for the LSTM model is the lowest, at 83.22%. I expect this is due to the smaller amount of data that is being used to train this neural network.

# In[58]:

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length = 150,
                                                          min_frequency = 20)
x_train_all_rnn = np.array(list(vocab_processor.fit_transform(clean_train_reviews_sentences)))
x_test_all_rnn = np.array(list(vocab_processor.transform(clean_test_reviews_sentences)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


# In[61]:

classifier_rnn = learn.Estimator(model_fn = rnn_model) 
classifier_rnn.fit(x_train_all_rnn, train.sentiment, steps=500) 

result_rnn = [p['class'] for p in classifier_rnn.predict(x_test_all_rnn, as_iterable=True)] 

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result_rnn})
output.to_csv("rnn_predictions.csv", index=False, quoting=3)


# Based on the testing data, it was not too surprising to see that the LSTM model also scored the worst on the submission data, 81.3%. It would be interesting to see how much the score would improve if we were able to use more data in the model.

# In[ ]:



