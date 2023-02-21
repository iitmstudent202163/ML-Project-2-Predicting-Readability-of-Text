# Predicting Readability of Texts Using Machine Learning
## 1.1 Introduction
There are a lot of books and magazines produced everyday that contain textual information. Sometimes, the difficulty of a text could be different based on the authors who have written it, the words used and also the sentence length and semantics. One of the challenges when reading texts from different sources is that the text might be too difficult for an average reader while it was originally intended for lecturers and scholars in a particular field. On the other hand, texts in a few sources might be too easy for some readers who expect it to have depth in its content. Therefore, different source of articles have different levels of difficulty in them.

In this machine learning project, we are going to be predicting the difficulty of texts based on some important features. Since we only have textual information, we are going to need to create new features and also tokenize sentences into different words to understand some useful features.

In addition to this, we would also find correlation between different features that we have created and see how much of an impact they make when we are performing the machine learning analysis and predictions.

## 1.2 Metrics
Mean Squared Error
Mean Absolute Error
## 1.3 Source

The data that is used is from Kaggle. There are many data sets available that could be used for machine learning purposes. Below is the link.

https://www.kaggle.com/c/commonlitreadabilityprize/data

# Table of Contents
## Predicting Readability of Texts Using Machine Learning
1.1 Introduction

1.2 Metrics

1.3 Source

## 2. Exploratory Data Analysis
2.1 Reading the Head of Training Data

2.2 Using Missing to plot the missing null values

2.3 Creating Useful Functions

2.4 Histogram of Target Variable

2.5 Histogram of Standard_Error Values

2.6 WordCloud function

2.7 Creating preprocessing function

## 3. Creating New Features
3.1 Defining get_useful_features function

3.2 Pairplot of Different Features

3.3 Heatmap of Correlation Matrix

3.4 Scatterplot function

3.5 Scatterplot between num_of_sentences and target

3.6 Scatterplot between preprocessed_essay_length and num_of_lemmas

3.7 Creating new function that generates more features

3.8 Pairplot of Other Features

3.9 Heatmap between num_of_quotes and num_of_questions

## 4. Creating New Functions
4.1 Creating Decontracted function

4.2 Creating preprocessing_function

4.3 Concatenating the dataframes

4.4 Creating min_max_scaler function

4.5 Creating Tfidf_Vectorizer function

4.6 Creating Count_Vectorizer function

4.7 Creating Vectorizer_decision function

4.8 Splitting the data into training and cross-validation data

## 5. Machine Learning Models
5.1 Neural Network Model

5.2 Tfidf Vectorizer

5.3 Creating actual_vs_predictions_dataframe

5.4 Creating regression_plot_function

5.5 Linear Regression Machine Learning Model

5.6 K - Neighbors Regressor

5.6.1 Regression Plot between Predicted Values and the Actual Values

5.7 PLS Regression

5.7.1 Regression Plot between Predicted Values and the Actual Values

5.8 Decision Tree Regressor

5.8.1 Regression plot between the Predicted Values and the Actual Values

5.9 Gradient Boosting Regressor

5.9.1 Regression plot between the Predicted Values and the Actual Values

## 6. Using Word2Vec
6.1 Importing Glove Vectors

6.2 Neural Network with Word2Vec

6.3 Defining the lineplot_neural_network function

6.4 Gradient Boosting Decision Tree for Average Word2Vec

6.5 Tfidf Word2Vec Featurization

6.6 Neural Network with Tfidf Word2Vec

6.7 Regression plot between the Predicted Values and the Actual Values


## 7. Conclusion
