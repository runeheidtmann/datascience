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
# fd = nltk.FreqDist(tokens)
# top_words = []
# for key, value in fd.items():
#     top_words.append((key, value))


# # sort the list by the top frequencies
# top_words = sorted(top_words, key = lambda x:x[1], reverse = True)

# # keep top 100 words only
# top_words = top_words[:100]

# top_word_series = pd.Series([w for (v,w) in top_words])
# top_word_series[:5]
# # get actual ranks of these words - wherever we see same frequencies, we give same rank
# word_ranks = top_word_series.rank(method = 'min', ascending = False)


# denominator = max(word_ranks)*min(top_word_series)

# # Y variable is the log of word ranks and X is the word frequency divided by the denominator
# # above
# Y = np.array(np.log(word_ranks))
# X = np.array(np.log(top_word_series/denominator))

# # fit a linear regression to these, we dont need the intercept!
# from sklearn import linear_model
# reg_model = linear_model.LinearRegression(fit_intercept = False)
# reg_model.fit(Y.reshape(-1,1), X)
# print("The value of theta obtained is:",reg_model.coef_)

# # make a plot of actual rank obtained vs theoretical rank expected
# plt.figure(figsize = (8,5))
# plt.scatter(Y, X, label = "Actual Rank vs Frequency")
# plt.title('Log(Rank) vs Log(Frequency/nx(n))')
# plt.xlabel('Log Rank')
# plt.ylabel('Log(Frequency/nx(n))')

# plt.plot(reg_model.predict(X.reshape(-1,1)), X, color = 'red', label = "Zipf's law")
# plt.legend()

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
