# Predicting Readability of Texts 
## 1.1 Introduction
There are a lot of books and magazines produced everyday that contain textual information. Sometimes, the difficulty of a text could be different based on the authors who have written it, the words used and also the sentence length and semantics. One of the challenges when reading texts from different sources is that the text might be too difficult for an average reader while it was originally intended for lecturers and scholars in a particular field. On the other hand, texts in a few sources might be too easy for some readers who expect it to have depth in its content. Therefore, different source of articles have different levels of difficulty in them.

In this machine learning project, we are going to be predicting the difficulty of texts based on some important features. Since we only have textual information, we are going to need to create new features and also tokenize sentences into different words to understand some useful features.

In addition to this, we would also find correlation between different features that we have created and see how much of an impact they make when we are performing the machine learning analysis and predictions.

## 1.2 Metrics
* Mean Squared Error
* Mean Absolute Error
## 1.3 Source

The data that is used is from Kaggle. There are many data sets available that could be used for machine learning purposes. Below is the link.

https://www.kaggle.com/c/commonlitreadabilityprize/data

# Machine Learning and Deep Learning

With machine learning and deep learning, it is possible to predict the readability of the text and understand some of the important features that determine the difficulty respectively.

Therefore, we have to consider a few important parameters when determining the difficulty of different machine learning models respectively.

We have to take into consideration the difficulty of the text along with other important features such as the number of syllables and the difficulty of the words in order to determine the overall level of the text.

# Natural Language Processing (NLP)

* We have to use the natural language processing (NLP) when we are dealing with the text respectively.
* Since we have a text, we have to use various processing techniques so that they are considered into forms that could be easy for machine learning   purposes.
* Once those values are converted into vectors, we are going to use them by giving them to different machine learning and deep learning models with a different set of layers respectively.
* We would be working with different machine learning and deep learning algorithms and understand some of the important metrics that are needed for the problem at hand.
* We see that since the target that we are going to be predicting is continuous, we are going to be using the regression machine learning techniques so that we get continuous output.

# Vectorizers

There are various vectorizers that were used to convert a given text into a form of a numeric vector representation so that it could be given to machine learning models for predictions for difficulty. Below are some of the vectorizers used to convert a given text to vectors.

* Count Vectorizer
* Tfidf Vectorizer
* Average Word2Vec (Glove Vectors)
* Tfidf Word2Vec

# Machine Learning Models

The output variable that we are considering is a continuous variable, therefore, we should be using regression techniques for predictions. Below are some of the machine learning and deep learning models used to predict the difficulty of texts.

* Deep Neural Networks
* Linear Regression
* K - Neighbors Regressor
* PLS Regression
* Decision Tree Regressor
* Gradient Boosting Regressor

# Outcomes

* TFIDF Word2Vec Vectorizer was the best encoding technique which results in a significant reduction in the mean absolute error respectively.
* Gradient Boosted Decision Trees (GBDT) were performing the best in terms of the mean absolute and mean squared error of predicting the difficulty of texts.
