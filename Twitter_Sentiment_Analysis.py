import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
#import jupyterthemes import jtplot
#jtplot.style(theme='monokai',,context = 'notebook',ticks = True , grid=False)

tweets_df= pd.read_csv('train.csv')

#tweets_df=tweets_df.drop(['id'],axis = 1)

sns.heatmap(tweets_df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
tweets_df.hist(bins=30,figsize=(13,5),color='r')
sns.countplot(tweets_df['label'],label='Count')
tweets_df['length']=tweets_df['tweet'].apply(len)
tweets_df['length'].plot(bins=100,kind='hist')

positive=tweets_df['tweet'][tweets_df['label']==0]
negative=tweets_df['tweet'][tweets_df['label']==1]

#sentences=tweets_df['tweet'].tolist()
#sentences_as_one=" ".join(sentences)
#plt.figure(figsize=(20,20))
#plt.imshow(wordcloud().generate(sentences_as_one))
#negative_list=negative['tweet'].tolist()
#negative_sentences_as_one=" ".join(negative_list)
#plt.figure(figsize=(20,20))


from wordcloud import WordCloud
positive_words =' '.join([text for text in positive])
wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Positive Words')
plt.show()
negative_words =' '.join([text for text in negative])
wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()

test_punc_removed=[]
#test=' Hi there beautiful peeps :). Have a great day'
import string
#for char in test:
#    if char not in string.punctuation:
#        test_punc_removed.append(char)
#test_joined=''.join(test_punc_removed)
from nltk.corpus import stopwords
stopwords.words('english')
#test_cleaned=[word for word in test_joined.split() if word.lower() not in stopwords.words('english')]




#sample = ['this is the 1st paper , this is second . and this is third.']
#vectorizer=CountVectorizer()
#X = vectorizer.fit_transform(sample)
#print(X.toarray())

def message_cleaning(message):
    test_punc_removed=[char for char in message if char not in string.punctuation]
    test_joined =''.join(test_punc_removed)
    test_cleaned = [word for word in test_joined.split() if word.lower() not in stopwords.words('english')]
    return test_cleaned


tweets_clean=tweets_df['tweet'].apply(message_cleaning)
vectorizer=CountVectorizer(analyzer=message_cleaning)
tweets_countvectorizer=CountVectorizer(analyzer= message_cleaning,dtype='uint8').fit_transform(tweets_df['tweet'])
tweet_V=tweets_countvectorizer.toarray()
X=tweet_V
y=tweets_df['label']


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_predict_test))
