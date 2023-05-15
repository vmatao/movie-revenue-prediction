#!/usr/bin/env python
# coding: utf-8

# # Data Science Challenge

# In[544]:


# If you'd like to install packages that aren't installed by default, uncomment the last two lines of this cell and replace <package list> with a list of your packages.
# This will ensure your notebook has all the dependencies and works everywhere

#import sys
#!{sys.executable} -m pip install <package list>


# In[4]:


#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 101)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from datetime import datetime


# ## Data Description

# Column | Description
# :---|:---------
# `title` |Title of the movie|
# `country` | Countries in which movie was released|
# `genres` | Movie Genres (Action ,Adventure, Comedy etc.)
# `language` | Languages in which movie was released
# `writer_count` | Number of writers of the movie
# `title_adaption` | Is movie original screenplay or adapted.
# `censor_rating` | Release rating given to the movie (R /PG-13/PG/NR/UR/G)
# `release_date` | Date when movie was released
# `runtime` | Movie runtime
# `dvd_release_date` | Date of release of DVD for sale
# `users_votes` | Number of users who voted for this movie to be included in Watch-It library
# `comments` | Number of comments on movie trailer(as of now)
# `likes` | Number of likes on movie trailer (as of now)
# `overall_views` | Number of views on movie trailer (as of now)
# `dislikes` | Number of dislikes on movie trailer (as of now)
# `ratings_imdb` | Rating given to movie on IMDB.
# `ratings_tomatoes` | Rating given to movie on Rotten tomatoes.
# `ratings_metacritic` | Rating given to movie on Metacritic etc.
# `special_award` | Number of awards nominations/winnings in BAFTA, Oscar or  Golden Globe.
# `awards_win` | awards won by the movie
# `awards_nomination` | Number of awards nominations
# `revenue_category` | Revenue Category (High/Low)

# ## Data Wrangling & Visualization

# In[45]:


# Dataset is already loaded below
train_data = pd.read_csv("train.csv")

# extract label and transofrm to np.array
y = train_data[["revenue_category"]].copy()
y_le = LabelEncoder()
y['revenue_category'] = y_le.fit_transform(y['revenue_category'])
y = y.iloc[:,:].values


# In[46]:


#Explore columns
train_data.columns


# In[47]:


train_data.dtypes


# In[48]:


def convert_to_float_imdb(x):
    return float(x.split('/')[0]) / 10

def convert_to_float_meta(x):
    return float(x.split('/')[0]) / 100

def convert_to_float_tomato(x):
    return float(x.split('%')[0]) / 100


# In[49]:


def prepare_data(data, test):
    
    # drop columns with probable no effect on the accuracy
    data = data.drop(['title','title_adaption'], axis=1)
    
    data.censor_rating = data.censor_rating.astype(str)
    data.writer_count = data.writer_count.astype('Int64')
    data['writer_count'] = data['writer_count'].fillna(0)

    le = LabelEncoder()
    data['censor_encoded'] = le.fit_transform(data['censor_rating'])
    
    # hot encode genres and language country
    encoded = pd.get_dummies(data['genres'].str.split(',\s+').explode()).sum(level=0)
    data = pd.concat([data, encoded], axis=1)

    encoded = pd.get_dummies(data['language'].str.split(',\s+').explode()).sum(level=0)
    data = pd.concat([data, encoded], axis=1)
    
    encoded = pd.get_dummies(data['country'].str.split(',\s+').explode()).sum(level=0)
    data = pd.concat([data, encoded], axis=1)
    
    # extract day month year from release date and dvd rel date
    data['release_date'] = data['release_date'].fillna('31-May-90')
    data['release_date']= pd.to_datetime(data['release_date'], format='%d-%b-%y')
    data['release_day']=data['release_date'].apply(lambda x:x.weekday())
    data['release_month']=data['release_date'].apply(lambda x:x.month)
    data['release_year']=data['release_date'].apply(lambda x:x.year)

    data['dvd_release_date'] = data['dvd_release_date'].fillna('31-May-90')
    data['dvd_release_date']= pd.to_datetime(data['dvd_release_date'], format='%d-%b-%y')
    data['dvd_release_day']=data['dvd_release_date'].apply(lambda x:x.weekday())
    data['dvd_release_month']=data['dvd_release_date'].apply(lambda x:x.month)
    data['dvd_release_year']=data['dvd_release_date'].apply(lambda x:x.year)
    
    # remove strings and turn into numeral
    data.runtime = data.runtime.str.replace(' min' , '')
    data.runtime = data.runtime.astype(int)

    data.users_votes = data.users_votes.str.replace(',' , '')
    data.users_votes = data.users_votes.astype(int)
    
    # fill empty with 0
    data.comments = data.comments.astype('Int64')
    data['comments'] = data['comments'].fillna(0)

    data.likes = data.likes.astype('Int64')
    data['likes'] = data['likes'].fillna(0)

    data.dislikes = data.dislikes.astype('Int64')
    data['dislikes'] = data['dislikes'].fillna(0)

    data.overall_views = data.overall_views.astype('Int64')
    data['overall_views'] = data['overall_views'].fillna(0)
    
    # convert ratings to float
    data['ratings_imdb'] = data['ratings_imdb'].apply(convert_to_float_imdb)
    data['ratings_metacritic'] = data['ratings_metacritic'].apply(convert_to_float_meta)
    data['ratings_tomatoes'] = data['ratings_tomatoes'].apply(convert_to_float_tomato)
    
    # remove label column if not in test mode
    if test==False:
        data = data.drop(['genres','language','censor_rating','release_date','dvd_release_date','country','revenue_category'], axis=1)

    else:
        data = data.drop(['genres','language','censor_rating','release_date','dvd_release_date','country'], axis=1)
    
    columns_to_scale = ['writer_count', 'runtime', 'users_votes', 'comments', 'likes',
                        'overall_views', 'dislikes','special_award','awards_win','awards_nomination']

    features = data[columns_to_scale]
    scaler = StandardScaler().fit(features.values)
    data[columns_to_scale] = scaler.transform(features.values)

    return data, data.columns


# In[13]:


# missing=data.isna().sum().sort_values(ascending=False)
# sns.barplot(missing[:8],missing[:8].index)
# plt.style.use('dark_background')
# plt.show()


# In[182]:


# columns_to_normalize = ['writer_count', 'runtime', 'users_votes', 'comments', 'likes',
#                         'overall_views', 'dislikes','special_award','awards_win','awards_nomination']

# for col in columns_to_normalize:
#       print(col)
#       x_array=[]
#       x_array=np.array(data[col].fillna(0))
#       X_norm=normalize([x_array])[0]
#       data[col]=X_norm


# In[50]:


train_df, columns = prepare_data(train_data, test=False)


# In[51]:


train_df


# ## Visualization, Modeling, Machine Learning
# 
# Can you build a model that can help them predict what titles would be suitable for licensing and identify how different features influence their decision? Please explain your findings effectively to technical and non-technical audiences using comments and visualizations, if appropriate.
# - **Build an optimized model that effectively solves the business problem.**
# - **The model would be evaluated on the basis of accuracy.**
# - **Read the test.csv file and prepare features for testing.**

# In[52]:


#Loading Test data
test_data=pd.read_csv('test.csv')
test_data.head()


# In[53]:


test_df,columns_test = prepare_data(test_data, test=True)


# In[54]:


# some genres or languages might not be in the test set - 
# make sure that they have the same columsn and column order
cols = test_df.columns.union(train_df.columns)


# In[55]:


test_df = test_df.reindex(columns=cols, fill_value=0)
train_df = train_df.reindex(columns=cols, fill_value=0)


# In[56]:


X = train_df.iloc[:,:].values
print(X.shape)


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[37]:


# Logistic regression


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[36]:


# KNeighborsClassifier


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


# Kernel SVM


# In[60]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


# Random forests


# In[64]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[27]:


y.shape


# In[34]:


X_test = test_df.iloc[:,:].values
print(X_test.shape)

# train with whole data
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)



# 
# 
# **The management wants to know what are the most important features for your model.  Can you tell them?**
# 
# > #### Task:
# - **Visualize the top 20 features and their feature importance.**
# 

# **Visualize the top 20 features and their feature importance.**

# In[35]:


(pd.Series(classifier.feature_importances_, index=cols)
   .nlargest(20)
   .plot(kind='barh')) 


# **Visualize all features and their feature importance.**

# In[37]:


importances = classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 40))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols[indices])
plt.xlabel('Relative Importance')


# > #### Task:
# - **Submit the predictions on the test dataset using your optimized model** <br/>
#     For each record in the test set (`test.csv`), you must predict the value of the `revenue_category` variable. You should submit a CSV file with a header row and one row per test entry. The file (submissions.csv) should have exactly 2 columns:
# 
# The file (`submissions.csv`) should have exactly 2 columns:
#    - **title**
#    - **revenue_category**

# In[38]:


#Loading Test data
test_data=pd.read_csv('test.csv')
test_data.head()


# In[39]:


submission_df = test_data['title'].copy()


# In[40]:


revemie_df = pd.DataFrame(y_le.inverse_transform(y_pred),columns=['revenue_category'])


# In[41]:


submission_df = pd.concat([submission_df,revemie_df], axis=1)


# In[42]:


submission_df


# In[43]:


#Submission
submission_df.to_csv('submissions.csv',index=False)


# ---
