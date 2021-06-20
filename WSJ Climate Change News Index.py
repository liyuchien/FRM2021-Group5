#!/usr/bin/env python
# coding: utf-8

# In[1]:


########### USER's INPUT #############

save_path = r'D:\FRM'

#######################################


# In[2]:


user_stopwords = ["last","united","state","usa","uk","figure","well","due","chapter","table","clim","ate","et",
                  "also","may","pp","al","le","http","et al","im","would","many","could","management","model","use",
                  "using","per","new","used","event","however","data", "however","associated", "high","low",
                  "information", "example","different", "year", "total","see","one","two","need"]


# In[3]:


import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


# In[59]:


def preprocessing(text_lines):
    
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    
    text = ' '.join(text_lines)
    lower_text = text.lower()
    text_tokens = nltk.word_tokenize(lower_text)
    text_filter = filter(str.isalnum,text_tokens)
    stopwords = set(stopwords.words('english'))
    filtered_tokens = [word for word in text_filter if not word in stopwords]
    wnl = WordNetLemmatizer()
    token_stem = [wnl.lemmatize(word) for word in filtered_tokens]
    filtered_by_user = [word for word in token_stem if not word in user_stopwords]
    adj_list = ['climate' if word == 'clim' else word if word == 'ate' else word for word in filtered_by_user]
    adj2_list = [word for word in adj_list if not any([char.isdigit() for char in word])]
    text_list = [word for word in adj2_list if len(word) != 1]
    pos_tokens = nltk.pos_tag(text_list,tagset='universal')
    n_tokens_1 = []
    n = ["NOUN","VERB","ADJ",'ADV']
    for token in pos_tokens:
        if token[1] in n:
            n_tokens_1+=[token[0]]
    
    return n_tokens_1


# In[4]:


import os 
dict_file = os.path.join(save_path,'dictionary.txt')

with open(dict_file, encoding="utf-8") as f:
    text_lines = f.readlines()
    
text_list = preprocessing(text_lines)


# In[5]:


import nltk
from nltk import ngrams

ngrams = ngrams(text_list, 2)
bigrams =[gram for gram in ngrams]

fdist_bi = nltk.FreqDist(bigrams)
bi_freq = {}
for term,value in fdist_bi.items():
    bi_freq[term[0]+' '+term[1]] = value
    
fdist_uni = nltk.FreqDist(text_list)
uni_freq = {}
for term,value in fdist_uni.items():
    uni_freq[term] = value
    
combine_freq = dict(uni_freq)
combine_freq.update(bi_freq)


# In[6]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(max_words = 200, width = 1800, height = 1200, background_color="white")
wc.generate_from_frequencies(combine_freq)

plt.axis('off')
plt.imshow(wc)
plt.show()
wc.to_file(os.path.join(save_path,'wordcloud.jpg'))


# In[39]:


import pandas as pd
import os
file = os.path.join(save_path,'new_WSJ_all.xlsx')
df = pd.read_excel(file)


# In[8]:


from datetime import *
target_list = [date(int(i.split('/')[0]),int(i.split('/')[1]),int(i.split('/')[2])) for i in df['Date']]
test = date(2013,1,1)
miss = []
while test < date(2016,6,2):
    if not test  in target_list:
        miss.append(test.strftime('%Y%m%d'))
    test = test + timedelta(days=1)
print(miss)


# In[ ]:


daily_news_count = [df.iloc[i].count() for i in range(len(df))]
df.insert(1, 'daily_news_count', daily_news_count)
print(df[['Date','daily_news_count']].to_string())


# In[44]:


df = df.sort_values(by=['Date'])
print(df[['Date','daily_news_count']].to_string())


# In[48]:


plt.figure(figsize=(20,5))
plt.plot(df['Date'], df['daily_news_count'])
plt.xticks([i for i in range(len(cosine_sim)) if i%100==0],[df['Date'].iloc[i] for i in range(len(df)) if i%100==0])


# In[62]:


text_list_WSJ = []
for row in range(len(df)):
    print(str(row+1)+'/'+str(len(df)))
    row_list = [x for x in list(df.loc[row]) if str(x) != 'nan'][2:]
    news_list = preprocessing(row_list)
    text = ' '.join(news_list)
    text_list_WSJ.append(text)


# In[64]:


with open(fr'{save_path}\text_list_WSJ.json', 'w') as fh:
    json.dump(text_list_WSJ, fh)


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer
import gc
gc.collect()
tf = TfidfVectorizer(ngram_range=(1, 1))
tfidf = tf.fit_transform(text_list_WSJ).toarray().tolist()


# In[70]:


query_string = ' '.join(text_list)
values = tf.transform([query_string]).toarray()[0].tolist()


# In[71]:


from scipy.spatial import distance

cosine_sim = [(1-distance.cosine(tfidf[doc],values))*10000 for doc in range(len(tfidf))]
date = [i[:-3] for i in list(df['Date'])]                     
index_df = pd.DataFrame({'Date':date,'cosine_sim':cosine_sim})
index_df = index_df.groupby(['Date']).mean()
monthly_cosine_sim = index_df['cosine_sim']
df = df.sort_values(by=['Date'])


# In[ ]:


daily_cosine_sim = pd.DataFrame({'Date':df['Date'], 'cosine_sim':cosine_sim})
daily_cosine_sim = daily_cosine_sim.set_index(['Date'])
daily_cosine_sim.to_excel('D:\FRM\daily_cosine_sim.xlsx')
index_df.to_excel(fr'{save_path}\monthly_cosine_sim.xlsx')


# In[72]:


import matplotlib.pyplot as plt
plt.figure(figsize=(100,5))
plt.plot(range(len(cosine_sim)), sklearn.preprocessing.scale(cosine_sim))
plt.plot(df['Date'], sklearn.preprocessing.scale(df['daily_news_count']))
plt.xticks([i for i in range(len(cosine_sim)) if i%100==0],[df['Date'].iloc[i] for i in range(len(df)) if i%100==0])
plt.title('Daily WSJ Climate News Index')
plt.xlabel('Date')
plt.ylabel('WSJ')
plt.show()


# In[73]:


from scipy.stats import pearsonr
pearsonr(cosine_sim, df['daily_news_count'])


# In[74]:


plt.figure(figsize=(10,5))
plt.plot(range(len(monthly_cosine_sim)), monthly_cosine_sim)
plt.xticks([i for i in range(len(monthly_cosine_sim)) if i%4==0],            [index_df.index[i] for i in range(len(monthly_cosine_sim)) if i%4==0])
plt.title('Monthly WSJ Climate News Index')
plt.xlabel('Date')
plt.ylabel('WSJ')
plt.show()
plt.savefig(os.path.join(save_path,'WSJ_index.png'))


# In[75]:


# Check for stationarity of the time-series data
# We will look for p-value. In case, p-value is less than 0.05, the time series
# data can said to have stationarity
#
from statsmodels.tsa.stattools import adfuller
#
# Run the test
#
df_stationarityTest = adfuller(cosine_sim, autolag='AIC')
#
# Check the value of p-value
#
print("P-value: ", df_stationarityTest[1])
#
# Next step is to find the order of AR model to be trained
# for this, we will plot partial autocorrelation plot to assess
# the direct effect of past data on future data
#
from statsmodels.graphics.tsaplots import plot_pacf
pacf = plot_pacf(cosine_sim, lags=25)
plt.xlabel('Lag')
plt.ylabel('Correlation')


# In[76]:


from statsmodels.tsa.ar_model import AutoReg
# Create training and test data
#
train_data = cosine_sim
#
# Instantiate and fit the AR model with training data
#
ar_model = AutoReg(train_data, lags=3).fit()
#
# Print Summary
#
print(ar_model.summary())


# In[18]:


residual = pd.Series(ar_model.resid, index=df['Date'][3:])
residual.to_excel(fr'{save_path}\daily_residual.xlsx')

