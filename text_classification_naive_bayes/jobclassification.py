#%%
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation


#Method that removes punctuation from a string.
def remove_punctuation(s):
    global punctuation
    for p in punctuation:
        s = s.replace(p, '')
        s = s.replace('¨','')
    return s

#Method that removes numbers from a string.
def remove_nums(s):
    return re.sub('[^\s]*[0-9]+[^\s]*', "", s)

#Method that removes irrelevant words defined beforehand in a list
#takes an array of words.
def remove_stop_words(s):
    with open('stopwords.txt') as f:
        stopwords = f.read().splitlines()
        wordsInS = s.split()
        s = " ".join([w for w in wordsInS if w not in stopwords])

        return s
                
        

# A method that finds the root of a danish word
def word_root(s):
   wdf = pd.read_csv('fuldformer.txt', delimiter=';', names=["root","inflection","gender"])

   root = wdf.loc[wdf['inflection']==s]
   
   if len(root):
       return root.iloc[0]['root']
   else:
       return s


###########
# Load data
df = pd.read_csv('jobtitles.csv',delimiter=',')

df['clean_title'] = df['job_title'].map(remove_punctuation)
df['clean_title'] = df['clean_title'].map(remove_nums)
df['clean_title'] = df['clean_title'].map(lambda x: x.lower())
df['clean_title'] = df['clean_title'].map(remove_stop_words)

corpus = " ".join(df['clean_title'].tolist())
tokens = corpus.split()

X = np.array(df.loc[:, 'clean_title'])
y = np.array(df.loc[:, 'job_category'])

# split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
nb_mult_model = MultinomialNB().fit(X_train_counts, y_train)
predicted = nb_mult_model.predict(X_test_counts)
from sklearn import metrics
print(classification_report(y_test, predicted))
plot_confusion_matrix(nb_mult_model, 
                      X_test_counts, y_test,
                      cmap=plt.cm.Blues,
                      xticks_rotation='vertical')
plt.tight_layout()
plt.show()


import pickle
s = pickle.dumps(nb_mult_model)
clf2 = pickle.loads(s)

d = ["IT Student Assistant hos IMPACT A/S"]
dat = pd.DataFrame(d, columns = ['job_title'])
dat['clean_title'] = dat['job_title'].map(remove_punctuation)
dat['clean_title'] = dat['clean_title'].map(remove_nums)
dat['clean_title'] = dat['clean_title'].map(lambda x: x.lower())
dat['clean_title'] = dat['clean_title'].map(remove_stop_words)
print(dat.head())

testers = count_vectorizer.transform(dat['clean_title'])
pop = clf2.predict(testers)
print(pop)
# %%


# %%
