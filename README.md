# Sentiment Analysis
### Sentiment analysis project using LSTMs and word embeddings in Tensorflow

#### Current Status
A "proof of concept" model has been built with my initial dataset for all of the implementations given below. Everything is working but I believe you will not see the models perform well until I use a much larger training set.

#### Data
The first dataset I'll use is from Kaggle. It is comprised of sentences from Yelp, Amazon, and IMDB reviews. 
Each sentence is considered to be either positive or negative--there are not supposed to be any neutral sentences. 
Sentiment is binary.

I built a neural network architecture to predict the positive or negative sentiment of each sentence. 
In the notebook above I tried a few approaches, but in all cases I used RNNs to take in variable length sequences and outout a probability of the sentence being positive. 
I experimented with different RNN cell types but settled on using LSTMs in all examples in the notebook.

Here were some different implementations I tried:
1. Encode each word with an ID and learn an embedding especially as the input to the LSTM.
2. Using pre-trained GloVe embeddings as the input to the LSTM
3. Using a birectional LSTM with lerned embeddings

#### Learned embeddings
I used Gensim to create a vocabulary which contained all the words in the data set that appeared in 5 or more of the sentences.
These I fed through an embedding layer (a matrix) that mapped them to 64 dimensional vectors (down from vocab size of > 800). 
These were fed into the recurrent layer.

#### Pre-trained GloVe embeddings
I used the GloVe word vectors from the spaCy library to convert each word to a 300 dimensional vector. These were fed into the recurrent layer.
The reason I tried this was the I was quite sure that these vectors would contain much more contextual information on a word that I would get in my tiny corpus.
These embeddings are trained on billions of tokens after all!

#### Birectional LSTM
I tried bidirectional LSTMs to see if performance improved. I used the same architecture and same number of cells for the forward and backward cells.

### Future
I'll train my model on a larger data set. I would like to have longer sequences of text and more examples. 
As it was there were only < 2000 training examples and the median sequence length was 9. 
Given that the vocab size was > 800 and that the document frequency of words was extremely skewed, it's likely many words in the vocab were simply not present in the test set. 
In addition many words that did appear in the training set may have only appeared a handful of times (under 10) after the test/train split. 
I believe more data and longer sentences would help me build a more generalizable model.

Build a comparison model using classical NLP appraoches: bag of words models with naive bayes or SVM classifiers.
