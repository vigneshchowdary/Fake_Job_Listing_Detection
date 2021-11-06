#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')


# In[2]:


get_ipython().system('pip install -U spacy')


# In[5]:


import re
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[7]:


df=pd.read_csv(r"C:\Users\Vignesh Chowdary\OneDrive\Documents\Downloads\archive (4)\fake_job_postings.csv")
df.head()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[13]:


columns=['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions', 'employment_type']
for colu in columns:
    del df[colu]


# In[14]:


df.head()


# In[15]:


df.fillna('',inplace=True)


# In[55]:


plt.figure(figsize=(10,3))
sns.countplot(y='fraudulent',data=df)
plt.show()


# In[17]:


df.groupby('fraudulent')['fraudulent'].count()


# In[19]:


exp = dict(df.required_experience.value_counts())
del exp['']


# In[20]:


exp


# In[58]:


plt.figure(figsize=(6,5))
sns.set_theme(style="whitegrid")
plt.bar(exp.keys(), exp.values())
plt.title("NUMBER OF JOBS WITH EXPERIENCE",size=29)
plt.xlabel('Experience', size=10)
plt.ylabel('No. of jobs', size=10)
plt.xticks(rotation=30)
plt.show()


# In[42]:


def split(location):
    l = location.split(',')
    return l[0]
df['country'] = df.location.apply(split)


# In[31]:


df.head()


# In[44]:


countr= dict(df.country.value_counts()[:20])
del countr['']
countr


# In[64]:


plt.figure(figsize=(6,5))
plt.title("COUNTRY-WISE JOB POSTING",size=29)
plt.bar(countr.keys(), countr.values())
plt.xlabel('Countries', size=7)
plt.ylabel('No. of jobs', size=10)
plt.xticks(rotation=20)
plt.show()


# In[53]:


edu = dict(df.required_education.value_counts()[:15])
del edu['']
edu


# In[63]:


plt.figure(figsize=(5,5))
plt.title("JOB POSTINGS BASED ON EDUCATION",size=29)
plt.bar(edu.keys(),edu.values())
plt.xlabel('No. of jobs', size=7)
plt.ylabel('Education', size=10)
plt.xticks(rotation=90)
plt.show()


# In[65]:


print(df[df.fraudulent==0].title.value_counts()[:10])


# In[66]:


print(df[df.fraudulent==1].title.value_counts()[:10])


# In[69]:


df['text']=df['title']+' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']
del df['country']


# In[73]:


df.head()


# In[74]:


fraudjobs_text = df[df.fraudulent==1].text
realjobs_text = df[df.fraudulent==0].text


# In[78]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3, max_words =3000 , width = 1600, height = 800 , stopwords =STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[79]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3, max_words =3000 , width = 1600, height = 800 , stopwords =STOPWORDS).generate(str(" ".join(realjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[80]:


pip install spacy && python -m spacy download en


# In[81]:


punctuations = string.punctuation

nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    
    mytokens = [ word.lemma_.lower().stop() if word.lemma_ !="-PRON-" else word.lower_ for word in mytokens ]
    
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    return mytokens

class predictors(TransformerMixin):
    def transform(self, x, **transform_params):
        
        return [clean_text(text) for text in x]
    
    def fit(self, x, y=None, **fit_params):
        return self
    
    def get_params(self, deeep=True):
        return {}
    
def clean_text(text):
    return text.strip().lower()


# In[82]:


df['text'] = df['text'].apply(clean_text)


# In[83]:


cv = TfidfVectorizer(max_features = 100)
x = cv.fit_transform(df['text'])
df1 = pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
df.drop(["text"], axis=1, inplace=True)
main_df = pd.concat([df1,df], axis=1)


# In[84]:


main_df.head()


# In[86]:


y = main_df.iloc[:,-1]
x = main_df.iloc[:,:-1]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[87]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
model=rfc.fit(x_train,y_train)


# In[88]:


print(x_test)


# In[89]:


pred = rfc.predict(x_test)
score = accuracy_score(y_test, pred)
score


# In[90]:


print("Classification Report\n")
print(classification_report(y_test, pred))
print("confusion Matrix\n")
print(confusion_matrix(y_test, pred))


# In[ ]:




