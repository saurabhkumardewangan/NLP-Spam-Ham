
################## Nlp project : Spam/ham text classification

############# Loading the libraries
import pandas as pd
import numpy as np

############# Load the dataset of SMS messages
df = pd.read_table('C:/Users/ADMIN/Desktop/NLP project/SMSSpamCollection', 
                   header=None, encoding='utf-8')

############# Information
print(df.info())

############# Head
print(df.head())

############# check class distribution
classes = df[0]            # Creating class labels column as classes

print(classes.value_counts()) 
## Imbalaced dataset but we don't to overpredict the spam so we leave it be


############# Pre processing
from sklearn.preprocessing import LabelEncoder

# convert class labels to binary values, 0 = ham and 1 = spam
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# checking the conversion
print(Y[:10])

print(classes[:10])


############## Storing the SMS message data as text messages
text_messages = df[1]
print(text_messages[:10])


############## Using regular expressions to replace email addresses, URLs, phone numbers, other numbers
# RegExLib.com
# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

# change words to lower case - Hello, HELLO, hello are all the same word
processed = processed.str.lower()

print(processed[:10])

################ Using stopwords
import nltk
nltk.download()
nltk.download("stopwords")

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
    
# removing stopwords from text messages

processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


################ Stemming

# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))

print(processed[:10])


############################### Feature Engineering ###########################
from nltk.tokenize import word_tokenize

# Creating bag-of-words models
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)


# Printing the total number of words
print('Number of words: {}'.format(len(all_words)))


# Printing and the 15 most common words
print('Most common words: {}'.format(all_words.most_common(15)))


# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]


# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(processed[5])
for key, value in features.items():
    if value == True:
        print (key)

processed[5]


############## Now lets do it for all the messages : find features
messages = list(zip(processed, Y))

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]

############################# Modeling #######################################

# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

print(len(training))
print(len(testing))



##################### We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# Training the model on the training data
model.train(training)

# Testing on the test dataset
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))                ### 98.7%

# Testing on the train dataset!
accuracy = nltk.classify.accuracy(model, training)*100
print("SVC Accuracy: {}".format(accuracy))               



##################### Other Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))

"""
K Nearest Neighbors Accuracy: 93.89806173725772
Decision Tree Accuracy: 97.4156496769562
Random Forest Accuracy: 98.20531227566404
Logistic Regression Accuracy: 98.7078248384781
SGD Classifier Accuracy: 98.49246231155779
Naive Bayes Accuracy: 97.84637473079684
SVM Linear Accuracy: 98.7078248384781

"""

####################### Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

print(models)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

"""
Voting Classifier: Accuracy: 98.7078248384781

"""

######################## Predicting class labels

# make class label prediction for testing set
txt_features, labels = zip(*testing)  # *unzipping testing data which has labels and text messages

prediction = nltk_ensemble.classify_many(txt_features)



######################## Confusion matrix
# print a confusion matrix and a classification report
print(classification_report(labels, prediction))

"""
                 precision    recall  f1-score   support

           0       0.99      1.00      0.99      1208
           1       0.97      0.91      0.95       185

    accuracy                           0.99      1393
   macro avg       0.99      0.95      0.97      1393
weighted avg       0.99      0.99      0.99      1393

"""

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])

"""
                  predicted     
                  ham  spam
actual ham       1206    2
       spam        17  168

"""
