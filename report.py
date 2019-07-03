import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re,nltk,string,pickle,random
from colorama import Fore,Back,Style,init
init()
from time import time
from matplotlib import pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
#col names for test data
col_names=['date','user_loc','followers','friends','message']
# read test csv file which was downloaded from twitter streamming API
df = pd.read_csv('data _190219-164056.csv', names=col_names, skiprows=1, header=None)
pd.set_option('display.max_colwidth',-1)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_columns", 100)
#text preprocessing (TEXT CLEANING)
def preprocess(text):
    # convert to lower case
    text1 = text.lower()
    # remove extra whitespaces
    text1 = re.sub('[\s]+', ' ', text1)
    # remove @ from username and replace with TWITTER_USER
    text1 = re.sub('@[^\s]+', '', text1)
    text1 = re.sub('[!@#$-]', '', text1)
    # remove http and www and replace with URL
    text1 = re.sub('((www\.[\s]+)|(https?://[^\s]+))', '', text1)
    # Replace #word with word Handling hashtags
    text1 = re.sub(r'#([^\s]+)', r'\1', text1)
    # strip
    text1 = text1.strip('\'"?!,.():;')
    # remove RT
    text1 = re.sub(r'\brt\b', '', text1)
    # remove 2+ dots
    text1 = re.sub(r'\.{2,}', ' ', text1)
    # remove underscore
    text1 = re.sub('\w(?<=_).*_?\w(?<=_)', '', text1)
    # emojis
    emoji = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\u200d"
                       u"\u2640-\u2642"
                       "]+", flags=re.UNICODE)
    text1 = emoji.sub(r'', text1)
    return text1
#data sends to preprocess function
df['message'] = df['message'].apply(preprocess)
# tokenize helper function
def text_process(raw_text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in list(raw_text) if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words('english')]


def remove_words(word_list):
    remove = ['tuesday', 'thoughts', '...', '“', '”', '’', '…', 'tuesdaythoughts']
    return [w for w in word_list if w not in remove]


# -------------------------------------------

# tokenize message column and create a column for tokens
df = df.copy()
df['tokens'] = df['message'].apply(text_process)  # tokenize style 1
df['no'] = df['tokens'].apply(remove_words)  # tokenize style 2
print(df.head())

model_NB = joblib.load("data/twitter_sentiment.pkl")
# append predictions to dataframe
tweet_preds = model_NB.predict(df['message'])
df_tweet_preds = df.copy()
df_tweet_preds['predictions'] = tweet_preds
print(df_tweet_preds.shape)
print(Back.MAGENTA+"-----------------------------------------------",Style.RESET_ALL)
index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.message[index],
                           df_tweet_preds.predictions[index]):
    print (Back.BLUE,sentiment,Style.RESET_ALL+ '--', text, '\n')
#df_tweet_preds.to_csv("final.csv")
# test result ___ 0=Negative, 4=Positive
pos = df_tweet_preds.predictions.value_counts()[4]
neg = df_tweet_preds.predictions.value_counts()[0]
print('Model predictions:')
print(Style.BRIGHT,Fore.GREEN+'Positives - {}'.format(pos),Style.RESET_ALL) 
print(Style.BRIGHT,Fore.RED+'Negatives - {}'.format(neg,pos),Style.RESET_ALL)
#function
def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    #grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)
    #sentiment_fit = grid.fit(X_train, y_train)
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print(Back.BLUE+"null accuracy: {0:.2f}%".format(null_accuracy*100),Style.RESET_ALL)
    print(Back.GREEN+"accuracy score: {0:.2f}%".format(accuracy*100),Style.RESET_ALL)
    if accuracy > null_accuracy:
        print(Back.RED+"model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100,Style.RESET_ALL))
    elif accuracy == null_accuracy:
        print(Back.RED+"model has the same accuracy with the null accuracy"+Style.RESET_ALL)
    else:
        print(Back.CYAN+"model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print(Style.RESET_ALL)
    print(Back.MAGENTA+"-"*80)
    print(Style.RESET_ALL)
    return accuracy, train_test_time
#end
df1 = pd.read_csv('final.csv')
names = ["Logistic Regression", "Linear SVM", "LinearSVM with L1-based feature selection","Multinomial NB",
         "Bernoulli NB","RandomForest","KMeans"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RandomForestClassifier(),
    KMeans()
    ]
zipped_clf = zip(names,classifiers)

X_train, X_test, y_train, y_test = train_test_split(df1['message'], df1['predictions'], test_size=0.2)
# create pipeline
result = []
parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }
for n,c in zipped_clf:
    pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words='english',
                            lowercase=True)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', c),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    print("Validation result for {}".format(n))
    print(c)
    clf_accuracy, tt_time = accuracy_summary(pipeline, X_train, y_train, X_test, y_test)
    result.append((n, clf_accuracy*100, tt_time))
    fin=pd.DataFrame(result)
print(Style.BRIGHT,Fore.CYAN,fin,"\n")
print(Style.RESET_ALL)


labels = fin[0]
sizes = fin[1]
colors = ['gold', 'green', 'lightcoral', 'blue','red','orange']
#explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=None, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

# this is where we define the values for GridSearchCV to iterate over
""""parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }"""
# do 10-fold cross validation for each of the 6 possible combinations of the above params
"""grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)
# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))
# save best model to current working directory
joblib.dump(grid, "data/final.pkl")
# load from file and predict using the best configs found in the CV step
model_NB = joblib.load("data/final.pkl" )

# get predictions from best model above
y_preds = model_NB.predict(X_test)

print('accuracy score: {0:.2f}%'.format(accuracy_score(y_test, y_preds)*100))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print("-"*80)
print("classification report")
print(classification_report(y_test, y_preds))"""