#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the pandas library
import pandas as pd


# In[3]:


# import Dataset 
df = pd.read_csv("C:\\Users\\gan8k\\OneDrive - Contoso\\Documents\\Kishore - Personal\\Data Science\\Recommendation Engine Project\\Gitb\\Revised Dataset - Restaurant.csv")


# In[4]:


# Getting the file information
df.info()

# Finding the null value in the column
df.isna().sum()


# In[5]:


# Importing the numpy library for numeric calculation
import numpy as np

# Importing the Simple Imputer to get the null value
from sklearn.impute import SimpleImputer


# In[6]:


# Mean Imputer for Numerical Data 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Rating"] = pd.DataFrame(mean_imputer.fit_transform(df[["Rating"]]))
df["Rating"].isna().sum()


# In[7]:


# Changing the numerical value to categorical value using Discretization

object = pd.cut(df.Age,bins = [0,16,30,38,45,60,99],labels=['Child','Young Aged Adults', 'Middle Aged Adults', 'Old Aged Adults','Old Aged','Elderly'])
object1 = pd.cut(df.TotalBill,bins = [0,1000,1500,2000],labels=['Low Fair','Medium Fair', 'High Fair'])
object2 = pd.cut(df.Rating,bins = [0,1,2,3,4,5],labels=['Worst','Poor', 'Average', 'Good','Excellent'])


# In[8]:


# Inserting a new column for the categorical value

df.insert(4,'AgeGroup', object)
df.insert(22,'BillGroup', object1)
df.insert(24,'RatingGroup', object2)


# In[9]:


#Checking the datatypes of the column value
df.dtypes


# In[10]:


# Changing the category data types to object
df = df.astype({"AgeGroup":'object',"BillGroup":'object',"RatingGroup":'object' })


# In[11]:


#Checking the datatypes of the column value
df.dtypes


# In[12]:


# Changing the Dataframe name to anime
anime = df

# Checking the shape (showing the number of rows & column)
anime.shape 

# Checking the column name (showing the column name)
anime.columns


# In[13]:


#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer 


# In[14]:


# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 


# In[15]:


# Preparing the Tfidf matrix by fitting and transforming
#Initiated all possible criteria based on Customer Details which be using to arrive the results.

tfidf_matrix = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients1']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix1 = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients2']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix2 = tfidf.fit_transform(anime['DiningType']+anime['Variety']+anime['AgeGroup']+anime['EmailID']+anime['Day']+anime['PreferrredIngredients']+anime['MainIngredients2']+anime['BillGroup']+anime['RatingGroup'])
tfidf_matrix.shape 


# In[16]:


#importing the linear kernel library
from sklearn.metrics.pairwise import linear_kernel


# In[17]:


# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
cosine_sim_matrix2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)


# In[18]:


# creating a mapping of anime name to index number 
anime_index = pd.Series(anime.index, index = anime['Order1']).drop_duplicates()
anime_index1 = pd.Series(anime.index, index = anime['Order2']).drop_duplicates()
anime_index2 = pd.Series(anime.index, index = anime['Order3']).drop_duplicates()


# In[19]:


#Checking whether the below code shows the column index no.
anime_id = anime_index["Gulab Jamun"]
anime_id


# In[20]:


# defing the recommendation function based on the Order Item1 using the cosine socre getting the pair wise similarity value to get the output

def get_recommendations(Order1, topN):    
    # topN = 10
    # Getting the Order1 index using its title 
    anime_id = anime_index[Order1]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order1 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the Order1 index 
    anime_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar Order1 and scores
    anime_similar_show = pd.DataFrame(columns=["Order1", "Score"])
    anime_similar_show["Order1"] = anime.loc[anime_idx, "Order1"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (anime_similar_show)
    return (anime_similar_show)



# defing the recommendation function based on the Order Item2

def get_recommendations1(Order2, topN):    
    # topN = 10
    # Getting the Order2 index using its title 
    anime_id1 = anime_index1[Order2]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores1 = list(enumerate(cosine_sim_matrix1[anime_id1]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores1 = sorted(cosine_scores1, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order2 
    cosine_scores_N1 = cosine_scores1[0: topN+1]
    
    # Getting the Order2 index 
    anime_idx1  =  [i[0] for i in cosine_scores_N1]
    anime_scores1 =  [i[1] for i in cosine_scores_N1]
    
    # Similar Order2 and scores
    anime_similar_show1 = pd.DataFrame(columns=["Order2", "Score"])
    anime_similar_show1["Order2"] = anime.loc[anime_idx1, "Order2"]
    anime_similar_show1["Score"] = anime_scores1
    anime_similar_show1.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (anime_similar_show1)
    # return (anime_similar_show)


# In[22]:


# defing the recommendation function based on the Order Item3

def get_recommendations2(Order3, topN):    
    # topN = 10
    # Getting the Order3 index using its title 
    anime_id2 = anime_index2[Order3]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores2 = list(enumerate(cosine_sim_matrix2[anime_id2]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores2 = sorted(cosine_scores2, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order3 
    cosine_scores_N2 = cosine_scores2[0: topN+1]
    
    # Getting the movie index 
    anime_idx2  =  [i[0] for i in cosine_scores_N2]
    anime_scores2 =  [i[1] for i in cosine_scores_N2]
    
    # Similar Order3 and scores
    anime_similar_show2 = pd.DataFrame(columns=["Order3", "Score"])
    anime_similar_show2["Order3"] = anime.loc[anime_idx2, "Order3"]
    anime_similar_show2["Score"] = anime_scores2
    anime_similar_show2.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (anime_similar_show2)
    # return (anime_similar_show)


# In[23]:


# Enter your Order Name and number of Order's to be recommended - This is based on Order1
get_recommendations("Risotto Lobster With Parmesan Egg Pancake, Confit Tomatoes And Coral Tuille", topN = 10)

# If we use the same order on order2 & Order3, the recommendation will be different, will not provide the same output.
# IF we recommend the same item it would not be correct, hence the suggestion will be different based on the order items.


# In[24]:


# Enter your Order Name and number of Order's to be recommended - This is based on Order2
get_recommendations1("Mutton Tahari", topN = 10)

# If we use the same order on order1 & Order3, the recommendation will be different, will not provide the same output.
# IF we recommend the same item it would not be correct, hence the suggestion will be different based on the order items.


# In[25]:


# Enter your Order Name and number of Order's to be recommended - This is based on Order3
get_recommendations2("Mango Phirni", topN = 10)

# If we use the same order on order1 & Order2, the recommendation will be different, will not provide the same output.
# IF we recommend the same item it would not be correct, hence the suggestion will be different based on the order items.


# In[26]:


# Food Recommended based on all criteria.. 
#(i.e Based on Customer Email ID, Age, Day, DiningType, Variety, Main Ingredients,Preferred Ingredients,Total Bill, Rating)

import pickle

pickle.dump(anime, open('Order1.pkl','wb'))

anime["Order1"].values

anime.to_dict()

pickle.dump(anime.to_dict(), open('Order1.pkl','wb'))

pickle.dump(cosine_sim_matrix, open('similarity.pkl','wb'))

anime.iloc[15].Order1


