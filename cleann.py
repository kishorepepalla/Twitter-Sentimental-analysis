import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from colorama import Fore,Back,Style,init
init()
import re,nltk,string,pickle,random
from matplotlib import pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
#from wordcloud import WordCloud
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

pd.set_option('display.max_colwidth',-1)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_columns", 100)

#train data exploration\
data = pd.read_csv("data/train.csv",error_bad_lines=False,encoding='latin1')
data.columns= ['label','id','date','flag','user','text']
data = data.drop(['id','user','flag','date'],axis=1)
# check the number of positive vs. negative tagged sentences
positives = data['label'][data.label == 4]
negatives = data['label'][data.label == 0]
neutrals  = data['label'][data.label == 2]

print(Style.BRIGHT,Fore.GREEN+'number of positve tagged sentences is:  {}'.format(len(positives)),Style.RESET_ALL)
print(Style.BRIGHT,Fore.RED+'number of negative tagged sentences is: {}'.format(len(negatives)),Style.RESET_ALL)
print('number of neutrals tagged sentences is: {}'.format(len(neutrals)))
print(Back.BLUE+'total length of the data is:            {}'.format(data.shape[0]),Style.RESET_ALL)
print(Back.MAGENTA+"---------------------------------------------------------------------------")
print(Style.RESET_ALL)
#get unique label count
#print(data.groupby('label').describe())
#word count
def word_count(tweet):
    return len(tweet.split())
data['word_count'] = data['text'].apply(word_count)
print(data.head(5))
print(Back.MAGENTA,"-------------------------------------------------------------------------------")
print(Style.RESET_ALL)
#plot word_count with pos and neg freq
x=data['word_count'][data.label == 4]
y=data['word_count'][data.label == 0]
plt.figure(figsize=(12,6))
plt.xlim(0,45)
plt.xlabel("word_count")
plt.ylabel("freq")
g=plt.hist([x,y],color=['r','b'],alpha=0.5,label=['positive','negative'])
plt.legend(loc='upper right')
plt.show()
#end to train data
#start test data
#col names for test data
col_names=['date','user_loc','followers','friends','message']
# read test csv file which was downloaded from twitter streamming API
df = pd.read_csv('data _190219-164056.csv', names=col_names, skiprows=1, header=None)

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
all_words = []
for line in list(df['message']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

# plot word frequency distribution of first few words
plt.figure(figsize=(12,5))
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(50,cumulative=False)
plt.show()
# log-log of all words
word_counts = sorted(Counter(all_words).values(), reverse=True)

plt.figure(figsize=(12,5))
plt.loglog(word_counts, linestyle='-', linewidth=1.5)
plt.ylabel("Freq")
plt.xlabel("Word Rank")
plt.show()

# show stop words examples
"""stop_words = stopwords.words('english') # Show some stop words
' , '.join(stop_words)[:200], len(stop_words)
print(stop_words)"""


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
print(Back.MAGENTA+'-'*80)
print(Style.RESET_ALL)


#wordcloud
"""all_words = []
for line in df['no']:  # try 'tokens'
    all_words.extend(line)

# create a word frequency dictionary
wordfreq = Counter(all_words)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17, 14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()"""
#------------------------------------------------------------------------------------
#vectorizing
bow = CountVectorizer(analyzer=text_process).fit(df['message'])
print(len(bow.vocabulary_))
print('\n')
print(Style.DIM,Fore.MAGENTA,bow.vocabulary_,Style.RESET_ALL)
bows=pd.Series(bow.vocabulary_).to_frame()
bows1=pd.DataFrame(bows)
print(Style.BRIGHT,Fore.CYAN,bows1)
print(Back.MAGENTA+"----------------------------------------------------------------------------")
print(Style.RESET_ALL)
# tranform entire dataframe of messages
messages_bow = bow.transform(df['message'])
# check out the bag-of-words counts for the entire corpus as a large sparse matrix
print('message_bow',messages_bow)
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
print(Back.MAGENTA+"------------------------------------------------------------------------------")
print(Style.RESET_ALL)
#TfidTranfoem starts
tfidftransformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidftransformer.transform(messages_bow)
print(messages_tfidf.shape)
#end test data
#----------------------------------------------------------
#piplene for ML
#pipline analyzer
#X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(data['text'][:200000], data['label'][:200000], test_size=0.2)
# create pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words='english',
                            lowercase=True)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
# this is where we define the values for GridSearchCV to iterate over
parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }
# do 10-fold cross validation for each of the 6 possible combinations of the above params
grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)
# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(Fore.YELLOW+"Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param),Style.RESET_ALL)
# save best model to current working directory
joblib.dump(grid, "data/twitter_sentiment.pkl")
# load from file and predict using the best configs found in the CV step
model_NB = joblib.load("data/twitter_sentiment.pkl" )

# get predictions from best model above
y_preds = model_NB.predict(X_test)

print(Style.BRIGHT,Fore.GREEN+'accuracy score: ',accuracy_score(y_test, y_preds),Style.RESET_ALL)
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))
print(Back.MAGENTA+"----------------------------------------------------------------------------------------------")
print(Style.RESET_ALL)
# run predictions on twitter data
tweet_preds = model_NB.predict(df['message'])

# append predictions to dataframe
df_tweet_preds = df.copy()
df_tweet_preds['predictions'] = tweet_preds
print(df_tweet_preds.shape)
print(Back.MAGENTA+"-----------------------------------------------")
print(Style.RESET_ALL)
print('final Data Output')
index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.message[index],
                           df_tweet_preds.predictions[index]):
    print (Back.BLUE,sentiment,Style.RESET_ALL+ '--', text, '\n')
df_tweet_preds.to_csv("final.csv")
# load model
"""model_NB = joblib.load("data/twitter_sentiment.pkl" )"""

# test string
"""sample_str =While ride-sharing first mover Uber has fallen on tough times with
scandal and abyssal track records of leadership, and cash burning
growth-orientated practices, the world has caught up with self-driving
tech with many players now in the race."""


"""p = model_NB.predict([sample_str])

# formatting helper
def sentiment_str(x):
    if x==0:
        return 'Negative'
    else:
        return 'Positive'
#_____________________________________________

# test result ___ 0=Negative, 1=Positive
print("the sentence: \n\n'{}' \n\nhas a {} sentiment".format(sample_str,sentiment_str(p[0])))"""
"""pos = df_tweet_preds.predictions.value_counts()[0]
neg = df_tweet_preds.predictions.value_counts()[1]

print('Model predictions: Positives - {}, Negatives - {}'.format(neg,pos))"""
