#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import chardet
with open('spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# In[3]:


df=pd.read_csv('spam.csv',encoding='Windows-1252')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


#DATA CLEANING
#EDA
#TEXT PREPROCESSING
#MODEL BUILDING
#EVALUATION
#IMPROVEMENTS
#CONVERTING INTO A WEBSITE


# In[7]:


##DATA CLEANING


# In[8]:


df.info()


# In[9]:


#we'll drop last three columns coz they seem unimportant ..have lesser values than the other two columns 

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.head()


# In[10]:


df.rename(columns={'v1':'Target','v2':'text'},inplace=True)
df.head()


# In[11]:


from sklearn.preprocessing import LabelEncoder
en=LabelEncoder()
df['Target']=en.fit_transform(df['Target'])
df.head()


# In[12]:


# checking for null values

df.isnull().sum()


# In[13]:


#checking for duplicate values

df.duplicated().sum()


# In[14]:


df=df.drop_duplicates(keep='first')
df.shape


# In[15]:


##EDA 


# In[16]:


df['Target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


plt.pie(df['Target'].value_counts(),labels=['not spam','spam'],autopct="%0.2f")
plt.show()


# In[19]:


#data is imbalanced 


# In[20]:


import nltk


# In[21]:


get_ipython().system('pip install nltk')


# In[22]:


nltk.download('punkt')


# In[23]:


##we'll find the number of characters,words,sentences

df['num_characters']=df['text'].apply(len)
df.head()


# In[24]:


#number of words

df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[25]:


df.head()


# In[26]:


df['text'].apply(lambda x:nltk.sent_tokenize(x))


# In[27]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df.describe()


# In[30]:


#ham messages analysis
df[df['Target']==0].describe()


# In[31]:


#spam messages analysis
df[df['Target']==1].describe()


# In[32]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['Target']==0]['num_characters'])
sns.histplot(df[df['Target']==1]['num_characters'],color='red')


# In[33]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['Target']==0]['num_words'])
sns.histplot(df[df['Target']==1]['num_words'],color='red')


# In[34]:


#finding corelation

sns.heatmap(df.corr(),annot=True)


# In[35]:


## DATA PREPROCESSING


# In[36]:


from nltk.corpus import stopwords
nltk.download('stopwords')


# In[37]:


#stemming
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('dancing')


# In[38]:


import string
string.punctuation


# In[39]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    for j in text:
        if(j not in stopwords.words('english') and j not in string.punctuation):
            y.append(ps.stem(j))
    
    return " ".join(y)
    


# In[40]:


transform_text('Alisha likes dancing?')


# In[41]:


df['transformed_text']=df['text'].apply(transform_text)


# In[42]:


df.head()


# In[43]:


#creating word cloud


# In[44]:


pip install wordcloud


# In[45]:


from wordcloud import WordCloud
wc=WordCloud(width=600,height=500,min_font_size=10,background_color='black')


# In[46]:


spam_wc=wc.generate(df[df['Target']==1]['transformed_text'].str.cat(sep=" "))


# In[47]:


plt.figure(figsize=(15,8))
plt.imshow(spam_wc)


# In[48]:


ham_wc=wc.generate(df[df['Target']==0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,8))
plt.imshow(ham_wc)


# In[49]:


spam_corpus=[]
for msg in df[df['Target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)


# In[50]:


from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(30))


# In[51]:


sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


ham_corpus=[]
for msg in df[df['Target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)


# In[53]:


from collections import Counter
pd.DataFrame(Counter(ham_corpus).most_common(30))


# In[54]:


sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# MODEL BUILDING

# In[55]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer()


# #using bag of words or frequent words logic
#     

# In[56]:


X=cv.fit_transform(df['transformed_text']).toarray()


# In[57]:


X.shape


# In[58]:


y=df['Target'].values


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[61]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[62]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[63]:


gnb.fit(X_train,Y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))


# In[64]:


mnb.fit(X_train,Y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(Y_test,y_pred2))
print(confusion_matrix(Y_test,y_pred2))
print(precision_score(Y_test,y_pred2))


# In[65]:


bnb.fit(X_train,Y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(Y_test,y_pred3))
print(confusion_matrix(Y_test,y_pred3))
print(precision_score(Y_test,y_pred3))


# using tfidf

# In[66]:


X=tfidf.fit_transform(df['transformed_text']).toarray()


# In[67]:


X.shape


# In[68]:


y=df['Target'].values


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[71]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[72]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[73]:


gnb.fit(X_train,Y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))


# In[74]:


mnb.fit(X_train,Y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(Y_test,y_pred2))
print(confusion_matrix(Y_test,y_pred2))
print(precision_score(Y_test,y_pred2))


# In[75]:


bnb.fit(X_train,Y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(Y_test,y_pred3))
print(confusion_matrix(Y_test,y_pred3))
print(precision_score(Y_test,y_pred3))


# In[ ]:





# In[ ]:




