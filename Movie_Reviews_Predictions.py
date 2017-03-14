
# coding: utf-8

# # Predicting the Sentiment of Movie Reviews

# The goal for this analysis is to predict if a review rates the movie positively or negatively. Inside this dataset there are 25,000 labeled movies reviews for training, 50,000 unlabeled reviews for training, and 25,000 reviews for testing. More information about the data can be found at: https://www.kaggle.com/c/word2vec-nlp-tutorial.
# 
# This data comes from the 2015 Kaggle competition, "Bag of Words Meets Bags of Popcorn." Despite the competition being finished, I thought it could still serve as a useful tool for my first Natural Lanugage Processing (NLP) project. Within this analysis you will find two methods for predicting the sentiment of movie reviews. I wanted to experiment with a couple of strategies to gain an understanding of different options and compare their results. The two methods that I will use are: 
# - Bag of Centroids with word2vec
# - TfidfVectorizer

# In[36]:

import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk.data
import logging  
import multiprocessing
import time
import tflearn
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import VotingClassifier as vc
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer 
from collections import defaultdict


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


# Let's take a look at each of the dataframes.

# In[5]:

train.head()


# In[6]:

train.review[0]


# In[7]:

unlabeled_train.head()


# In[8]:

unlabeled_train.review[0]


# In[9]:

test.head()


# In[10]:

test.review[0]


# Everything looks good with the data. No surprises so far.

# ## Model #1: Bag of Centroids

# In[11]:

def review_to_wordlist(review, remove_stopwords = True, stem_words = True):
    # Clean the text, with the option to remove stopwords and stem words.

    # Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    
    # Convert words to lower case and split them
    review_text = review_text.lower()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        words = review_text.split()
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", review_text)
    review_text = re.sub(r"it's", " it is", review_text)
    review_text = re.sub(r"that's", " that is", review_text)
    review_text = re.sub(r"\'s", " 's", review_text)
    review_text = re.sub(r"\'ve", " have", review_text)
    review_text = re.sub(r"won't", " will not", review_text)
    review_text = re.sub(r"don't", " do not", review_text)
    review_text = re.sub(r"can't", " can not", review_text)
    review_text = re.sub(r"cannot", " can not", review_text)
    review_text = re.sub(r"n\'t", " n\'t", review_text)
    review_text = re.sub(r"\'re", " are", review_text)
    review_text = re.sub(r"\'d", " would", review_text)
    review_text = re.sub(r"\'ll", " will", review_text)
    review_text = re.sub(r"!", " ! ", review_text)
    review_text = re.sub(r"\?", " ? ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    # Shorten words to their stems
    if stem_words:
        words = review_text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in words]
        review_text = " ".join(stemmed_words)
    
    # Return a list of words, with each word as its own string
    return review_text.split()


# In[12]:

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    # Split a review into parsed sentences
    # Returns a list of sentences, where each sentence is a list of words
    
    # Use the NLTK tokenizer to split the review into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence))
    
    # Return the list of sentences
    # Each sentence is a list of words, so this returns a list of lists
    return sentences


# In[13]:

sentences = [] 

print ("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# In[14]:

# Check how many sentences we have in total 
print (len(sentences))
print()
print (sentences[0])
print()
print (sentences[1])


# In[18]:

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 300      # Word vector dimensionality                      
min_word_count = 5      # Minimum word count                        
num_workers = 1         # Number of threads to run in parallel
context = 20            # Context window size                                                                                    
downsampling = 1e-4     # Downsample setting for frequent words

from gensim.models import word2vec

# Initialize and train the model
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
model_name = "{}features_{}minwords_{}context".format(num_features,min_word_count,context)
model.save(model_name)


# In[19]:

# Load the model, if necessary
# model = Word2Vec.load("300features_5minwords_20context") 


# In[20]:

# Take a look at the performance of the model
print(model.most_similar("man"))


# In[21]:

model.most_similar("great")


# In[22]:

model.most_similar("terribl")


# In[23]:

model.most_similar("movi")


# In[24]:

model.most_similar("best")


# The model looks good so far. Each of these words has appropriate similar words.

# In[25]:

model.syn0.shape


# In[26]:

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0 
num_clusters = int(word_vectors.shape[0] / 5)

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters,
                           n_init = 5,
                           verbose = 2)
idx = kmeans_clustering.fit_predict(word_vectors)


# In[27]:

# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number                                                                                            
word_centroid_map = dict(zip(model.wv.index2word, idx))


# In[28]:

# Clean the training and testing reviews.
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review))
    
print("Training reviews are clean")  

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review))
    
print("Testing reviews are clean") 


# In[29]:

clean_train_reviews[0]


# In[31]:

clean_test_reviews[0]


# In[32]:

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


# In[34]:

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


# Let's use GridSearchCV to find the optimal parameters for each classifier.

# In[39]:

def use_GridSearch(model, model_paramters, x_values):
    '''Find the optimal parameters for a model'''
    grid = GridSearchCV(model, model_paramters, scoring = 'roc_auc')
    grid.fit(x_values, train.sentiment)

    print("Best grid score = ", grid.best_score_)
    print("Best Parameters = ", grid.best_params_)


# In[111]:

# RandomForect Classifier
rfc_parameters = {'n_estimators':[100,200,300],
                  'max_depth':[3,5,7,None],
                  'min_samples_leaf': [1,2,3]}

rfc_model = rfc()

use_GridSearch(rfc_model, rfc_parameters, train_centroids)


# In[48]:

# Logistic Regression
lr_parameters = {'C':[0.005,0.01,0.05],
                 'max_iter':[4,5,6],
                 'fit_intercept': [True]}

lr_model = lr()

use_GridSearch(lr_model, lr_parameters, train_centroids)


# In[49]:

# Stochastic Gradient Descent Classifier 
sgd_parameters = {'loss': ['log'],
                  'penalty': ['l1','l2','none']}

sgd_model = sgd()

use_GridSearch(sgd_model, sgd_parameters, train_centroids)


# Let's double check the quality of the classifiers with cross validation, then train them.

# In[51]:

def use_model(model, x_values):
    '''
    Test the quality of a model using cross validation
    Train the model with the x_values
    '''
    scores = cross_val_score(model, x_values, train.sentiment, cv = 5, scoring = 'roc_auc')
    model.fit(x_values, train.sentiment)
    mean_score = round(np.mean(scores) * 100,2) 

    print(scores)
    print()
    print("Mean score = {}".format(mean_score))


# In[105]:

rfc_model = rfc(n_estimators = 300,
                max_depth = None,
                min_samples_leaf = 1)

use_model(rfc_model, train_centroids)


# In[106]:

lr_model = lr(C = 0.01,
              max_iter = 5,
              fit_intercept = True)

use_model(lr_model, train_centroids)


# In[107]:

sgd_model = sgd(loss = 'log',
                penalty = 'l1')

use_model(sgd_model, train_centroids)


# In[108]:

rfc_result = rfc_model.predict(test_centroids)
lr_result = lr_model.predict(test_centroids)
sgd_result = sgd_model.predict(test_centroids)

avg_result = (lr_result + rfc_result + sgd_result) / 3

avg_result_final = []
for result in avg_result:
    if result > 0.5:
        avg_result_final.append(1)
    else:
        avg_result_final.append(0)
        
avg_output = pd.DataFrame(data={"id":test["id"], "sentiment":avg_result_final})
avg_output.to_csv("avg_centroids_submission.csv", index=False, quoting=3)


# In[57]:

# Take a look at the submission file
avg_output[0:10]


# When I submit the results to the Kaggle competition its score (area under the ROC curve) is 0.880, which ranks 266/578, top 46%.

# ## Model 2: TfidfVectorizer

# In[96]:

# Count the number of different words in the reviews
word_counts = defaultdict(int)

for comment in clean_train_reviews:
    word_counts[" ".join(comment)] += 1

for comment in clean_test_reviews:
    word_counts[" ".join(comment)] += 1
print(len(word_counts))


# In[97]:

# Set the parameters for vectorizing the words in the reviews.
vectorizer = TfidfVectorizer(max_features = len(word_counts), 
                             ngram_range = (1, 3), 
                             sublinear_tf = True)


# In[60]:

# Join the words of the reviews.
# The list of lists becomes just a list of strings (strings = reviews).
clean_train_reviews_join = []
for review in clean_train_reviews:
    clean_train_reviews_join.append(" ".join(review))

clean_test_reviews_join = []
for review in clean_test_reviews:
    clean_test_reviews_join.append(" ".join(review))


# In[99]:

# Train the vectorizer on the vocabulary and convert reviews into matrices.
x_train_vec = vectorizer.fit_transform(clean_train_reviews_join)
print("x_train_vec is complete.")
x_test_vec = vectorizer.transform(clean_test_reviews_join)
print("x_test_vec is complete.")


# Use GridSearchcv to find the best parameters, just like with Method 1.

# In[78]:

rfc_parameters_vec = {'n_estimators':[100,200,300],
                      'max_depth':[3,5,7,None],
                      'min_samples_leaf': [1,2,3,4]}

rfc_model_vec = rfc()

use_GridSearch(rfc_model_vec, rfc_parameters_vec, x_train_vec)


# In[71]:

lr_parameters_vec = {'C':[5,6,7],
                 'max_iter':[1,2,3],
                 'fit_intercept': [True,False]}

lr_model_vec = lr()

use_GridSearch(lr_model_vec, lr_parameters_vec, x_train_vec)


# In[68]:

sgd_parameters_vec = {'loss': ['log'],
                  'penalty': ['l1','l2','none']}

sgd_model_vec = sgd()

use_GridSearch(sgd_model_vec, sgd_parameters_vec, x_train_vec)


# Double check the quality of the classifiers with cross validation, then train them.

# In[100]:

rfc_model_vec = rfc(n_estimators = 200,
                max_depth = None,
                min_samples_leaf = 3)

use_model(rfc_model_vec, x_train_vec)


# In[101]:

lr_model_vec = lr(C = 6,
              max_iter = 2,
              fit_intercept = False)

use_model(lr_model_vec, x_train_vec)


# In[102]:

sgd_model_vec = sgd(loss = 'log',
                penalty = 'none')

use_model(sgd_model_vec, x_train_vec)


# In[103]:

lr_result_vec = lr_model_vec.predict(x_test_vec)
rfc_result_vec = rfc_model_vec.predict(x_test_vec)
sgd_result_vec = sgd_model_vec.predict(x_test_vec)

avg_result_vec = (lr_result_vec + rfc_result_vec + sgd_result_vec) / 3

avg_result_final_vec = []
for result in avg_result_vec:
    if result > 0.5:
        avg_result_final_vec.append(1)
    else:
        avg_result_final_vec.append(0)
        
avg_output_vec = pd.DataFrame(data={"id":test["id"], "sentiment":avg_result_final_vec})
avg_output_vec.to_csv("avg_vec_submission1.csv", index=False, quoting=3)


# In[104]:

avg_output_vec[0:10]


# This method scores slightly higher, 0.895, which ranks 251/578, top 44%.

# Just for fun, let's see what happens when we combine all six predictions.

# In[109]:

avg_result_combine = (lr_result + rfc_result + sgd_result +
                      lr_result_vec + rfc_result_vec + sgd_result_vec) / 6

avg_result_final_combine = []
for result in avg_result_combine:
    if result > 0.5:
        avg_result_final_combine.append(1)
    else:
        avg_result_final_combine.append(0)
        
avg_output_combine = pd.DataFrame(data={"id":test["id"], "sentiment":avg_result_final_combine})
avg_output_combine.to_csv("avg_combine_submission.csv", index=False, quoting=3)


# In[110]:

avg_output_combine[0:10]


# This 'combined' submission scored inbetween method 1 and 2, 0.892. I expected that this ensemble strategy would have scored better than the two previous methods, but unfortunately, it did not.

# ## Summary

# The best performing tutorial from Google scores 0.845, which is the "Word2Vec - Bag of Centroids" example. I am pleased that I have improved upon this example, and built another method that scores even higher. Although I did not score at the top of the leaderboard, I am still pleased with my results and have learned a great deal. One thing that I will focus on with further projects of this nature is reducing the amount of overfitting. As you probably noticed, my models perform much better on the training data than the testing data. If I find some useful strategies online for producing models that generalize better, I'll try to return to this code to improve its results.
