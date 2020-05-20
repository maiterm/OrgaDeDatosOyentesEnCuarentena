import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice

train = pd.read_csv("train.csv")
#cleaning the keywords
train["keyword"]=train["keyword"].str.replace(r"\%20disaster","").str.replace("%20","")
unique_keyword=list(train["keyword"].unique())[1:]
keyword=[]
for word1 in unique_keyword:
    for word2 in unique_keyword:
        if word1[:5]==word2[:5]:
            if len(word1)<len(word2):
                keyword.append(word1)
for word in keyword:
    pattern = "^"+word[:5]
    col_word= [word]*(train["keyword"].str.contains(pattern).sum())
    train.loc[train["keyword"].str.contains(pattern, na=False),"keyword"]= col_word

#cleaning locations
print("Location not null")
print(train["location"].notnull().sum())
#new col to target the location,wich one seems real
train["location-target"]=True

train["location"]=train["location"].str.replace(r"[ .]$","")


#location with the word "where" are Fake
bool_where=train["location"].str.contains(r"[Ww]here",na=False)
train.loc[bool_where,"location-target"]= [False]*(bool_where.sum())
#print(train.loc[bool_where,"location"].str.contains("\d",na=False).sum())

#location digit 

bool_digit=train["location"].str.contains(r"[0-9]",na=False)
train.loc[bool_digit,"location-target"]= [False]*(bool_digit.sum())

#location simbols 
bool_sim=train["location"].str.contains(r"[\\!$%&@#/[\]()?*+]",na=False)
train.loc[bool_sim,"location-target"]= [False]*(bool_sim.sum())

#location simbols 
bool_words=train["location"].str.contains(r"\b[Ii]nternet\b|\b[hH]ome\b|\b[Ll]ive\b|\b[Oo]ur\b|\b[Mm]oon\b|\b[hH]ell\b|\b[gG]od\b|\b[Yy]ou\b|\b[Yy]our\b|\b[Ww]e\b|\b[Mm]y\b|\b[Ww]orld\b",na=False)
train.loc[bool_words,"location-target"]= [False]*(bool_words.sum())
#print(train.loc[bool_words,"location"])



#Some locations have a "," the place, and other bigger
bool_len=train["location"].str.split(",|-")
#print(train[bool_len,"location"])

train.loc[train["location-target"],"location1"] = train["location"].str.split(",|-").str[-2]
train.loc[train["location-target"],"location2"] = train["location"].str.split(",|-").str[-1]
print("more than 3 words in locations2")
print(train.loc[train["location2"].str.split().str.len()>3,"location2"].unique())
train.loc[train["location2"].str.split().str.len()>3,"location2"]=None

#print(train.loc[train["location2"].notnull(),["location","location2"]])

print("locations2 of 3 words")
print(train.loc[train["location2"].str.split().str.len()==3,"location2"].unique())
print("Less than 3 words")
print(train.loc[train["location2"].str.split().str.len()<3,"location2"].unique()[100:900])