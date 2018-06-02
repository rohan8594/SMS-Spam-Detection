## 1. Introduction

SMS Spam is defined as unwanted text messages that are often used for spreading unwanted ads.
Spam messages often contain malicious links and are notoriously used by hackers to steal user
information. To add to this, spam messages in some countries prove to be a major problem because
they often incur an additional cost. This, along with the lack of major mobile phone spam filtering
software is my motivation behind exploring the problem of SMS spam detection through the use of
data mining and machine learning techniques.

Throughout the course of this project, my main goal has been to find knowledge in data through the
KDD process by first collecting, analysing and understanding the data, then applying pre-processing
techniques to transform the data in a way that can be understood by a machine learning algorithm,
then applying classification algorithms to make predictions on the data, and finally using sound
evaluation strategies to evaluate and further improve my results.

## 2. Data

For this project two different text message datasets have been used. The first dataset is the SMS Spam
Collection Data Set that has been collected from the popular UCI Machine Learning Repository
([https://archive.ics.uci.edu/ml/datasets/sms+spam+collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)). This dataset contains a total of 5572
SMS messages out of which 4825 are non-spam (ham) messages and 747 are spam messages. The
dataset file contains one message per line. Each line is composed of two columns: one with label (ham
or spam) and other with the raw text. Here are some examples:

_ham What you doing? how are you?_

_ham Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he
started guessing who i was wif n he finally guessed darren lor._

_spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your
phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop_

The second dataset is a spam message dataset that has been scraped from Dublin Institute of
Technology’s website. This dataset contains only spam messages and has a total of 1353 spam
messages.

The primary dataset of the project is the UCI dataset which has been used for the most part of the
project for model building and evaluation. The Dublin dataset is simply used in the final stages when
we use it to test the bias in our learned classifiers.

## 3. Methodology / Main Steps

The following section gives a brief description of the methodology and main steps that have been
followed in this project.

### 3.1 Data Analysis and Feature Engineering

The first step involved performing exploratory data analysis on the dataset with the aim of analysing
and understanding the data better. I used python’s pandas and matplotlib library to do some basic
analysis and data visualization. One of the major findings in this step was that message length was
found to be somewhat correlated to the label of the message. The figure below shows the distribution
of the message length. As the figure shows, there are two distinct peaks, the first one is for ham 
messages that were found to have a shorter length than the spam messages which are represented by
the second peak. As a result, an important step in feature engineering was to add message length as a
new feature in the dataset. Another important step was organizing the entire message dataset into a
pandas dataframe. This made data manipulation a lot easier for the entire course of the project.

### 3.2 Text Pre-processing and Transformation

The next step was to apply some basic text pre-processing techniques to the data to clean the text
messages. There were three major things that were done, punctuation removal, stopwords removal
and stemming. The result of this step is a list of clean word tokens that are stored in a new
processed_msgs.csv file in the output folder of the project directory. These word tokens are then used
to convert each message into a ‘bag of words’ and then into a TF-IDF vector representation. The
message length feature is then appended to this TF-IDF matrix to produce the final feature vector
which I later fit into my machine learning classifier.

### 3.3 Text Classification

Once I have my feature vector, consisting of the TF-IDF scores for each message and the length
feature, I split it into a training and test set using a 70:30 split. The training set is used to train various
classifiers, namely, SVM, Multinomial Naïve Bayes, Decision Tree, KNN, Random Forest, AdaBoost
and Bagging Classifier. The test set is then used to make predictions. Once I have the predictions, I
compare the classifiers based on accuracy, f1-measure, precision and recall to get an initial idea of
how each classifier performs. In this step I also generate a file called misclassified_msgs.txt in the
output folder, that contains the list of all misclassified text messages for each classifier.

### 3.4 Parameter Tuning

Next, I apply parameter tuning to try and further improve the results of my learned classifiers. I
achieve this by using scikit-learn's GridSearchCV to perform an exhaustive grid search. Exhaustive
grid search is a way to select the best model out of a family of models by performing K-fold cross-
validation and tuning the model over a grid of model parameters.


### 3.5 K-Fold Cross Validation and Learning Curve

Next, I apply 5-fold cross validation and plot a learning curve to evaluate my learned classifier. The
learning curve is generated by plotting the error vs. the training set size (i.e., how better does the
model get at predicting the target as you the increase number of instances used to train it). It is
important to note that in the learning curve, there are two error scores that are monitored: one for the
validation set, and one for the training sets. We plot the evolution of the two error scores as training
sets change and end up with two curves. The evolution of the two curves over the training set size
gives us a good indication of bias and variance in our model and thereby diagnose whether our
classifier shows any underfitting or overfitting symptoms.

### 3.6 Test learned classifier on 2nd Dataset to check for bias

In the next step, I use the learned classifiers from the previous steps to test how they perform on my
second dataset that has been scraped from Dublin Institute of Technology’s website. In this step, I re-
train my classifiers using previously learned model parameters on the entire UCI message dataset. The
Dublin dataset is used simply as the test set and not used for training purpose here. If accuracy of
predictions on the Dublin dataset are close to previously achieved accuracies, then we know our
model doesn’t suffer from bias.

### 3.7 Addressing Imbalance of UCI Dataset

In the final step of my project, I try to address the imbalance of the UCI message dataset. Originally,
the UCI dataset has 4825 (86.6%) ham and 747 (13.4%) spam messages, which seems like a rather
big difference. Hence, in this step I cut down the UCI message dataset to 747 spam messages and
1000 ham messages to produce a more even playing field. Once, I have a more balanced dataset, the
classifiers are re-trained to check if changing the composition of the dataset has any impact on the
final results.

## 4. Instructions for Compiling

This project has been developed using Python3 and hence you will need Python3 installed on your
system to compile and run this project. The project folder contains a total of seven python scripts
namely, read_data.py, text_preprocessing.py, text_classification.py, parameter_tuning.py,
learning_curve.py, check_bias.py, and address_imbalance.py that must be compiled to produce the
final results. The datafiles are present in the folder named ‘smsspamcollection’. Here there are two
files, SMSSpamCollection that contains the UCI dataset and spam.xml that contain the Dublin
dataset. The output folder contains certain output files that are generated by running the programs.
The step-wise instructions to compile and run the program are given below.

a) First install all dependencies.

```
$ pip install numpy

$ pip install pandas

$ pip install nltk

$ pip install scipy

$ pip install matplotlib

$ pip install scikit-learn
```

b) Once all dependencies are installed you can navigate to the project directory on your terminal. The
first script to execute is read_data.py. This script only reads the UCI message dataset file and prints
the first 100 messages to give you an idea of what the data looks like.

```
$ python3 read_data.py
```

c) Next you can run the script containing the text pre-processing module.

```
$ python3 text_preprocessing.py
```

d) Next you can run the script containing the initial text classification module.

```
$ python3 text_classification.py
```

e) Next you can run the parameter tuning module.

```
$ python3 parameter_tuning.py
```
**NOTE: Parameter tuning on SVM takes a very long time (approx. 1 hour) and hence has been commented out on line 81 of the script. The above script will hence only produce parameter tuning results for Multinomial NB. But if you want to see the results for SVM and time
permits please uncomment line 81 and execute.**

f) Next you can run the script that contains the k-fold cross validation and learning curve modules.

```
$ python3 learning_curve.py
```

g) Next you can run the script that tests the learned classifier on the Dublin dataset to check for bias.

```
$ python3 check_bias.py
```

h) Finally, you can run the final script that addresses the imbalance of the UCI dataset.

```
$ python3 address_imbalance.py
```
