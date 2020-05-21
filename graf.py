import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice
import re

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
train["keyword"]=train["keyword"].str.replace("dead","death")

#cleaning locations

#new col to target the location,wich one seems real
train["location-target"]=True

train["location"]=train["location"].str.replace(r"[ .]$","")


#location with the word "where" are Fake
bool_where=train["location"].str.contains(r"[Ww]here",na=False)
train.loc[bool_where,"location-target"]= False
#print(train.loc[bool_where,"location"].str.contains("\d",na=False).sum())

#location digit 

bool_digit=train["location"].str.contains(r"[0-9]",na=False)
train.loc[bool_digit,"location-target"]= False

#location simbols 
bool_sim=train["location"].str.contains(r"[\\!$%&@#/[\]()?*+]",na=False)
train.loc[bool_sim,"location-target"]= False

#location simbols 
bool_words=train["location"].str.contains(r"\b[Ii]nternet\b|\b[Ff]ar\b|\b[Bb]each\b|\bWORLD\b|\b[hH]ome\b|\b[Ll]ive\b|\b[Oo]ur\b|\b[Mm]oon\b|\b[hH]ell\b|\b[gG]od\b|\b[Yy]ou\b|\b[Yy]our\b|\b[Ww]e\b|\b[Mm]y\b|\b[Ww]orld\b|\b[Ll]ife\b",na=False)
train.loc[bool_words,"location-target"]= False
#print(train.loc[bool_words,"location"])

#Some locations have a "," the place, and other bigger
train.loc[train["location-target"],"location1"] = train["location"].str.split(",|-").str[-2]
train.loc[train["location-target"],"location2"] = train["location"].str.split(",|-").str[-1]
#print("more than 3 words in locations2")
#print(train.loc[train["location2"].str.split().str.len()>3,"location2"].unique())

#There are a lot new york city
bool_ny=train["location2"].str.contains("New York",na=False)
bool_len=train["location2"].str.split().str.len()>=3
bool_ny_len= (bool_ny) & (bool_len)
train.loc[bool_ny_len,"location2"]="New York"
#print(train["location2"].str.contains("New York",na=False).sum())
#
#Descart long locations
train.loc[bool_len,"location-target"]=False
train.loc[bool_len,"location2"]=None
train["len_text"]=train["text"].str.len()

#print(train.loc[train["location2"].notnull(),["location","location2"]])
#print("locations2 of 3 words")
#print(train.loc[train["location2"].str.split().str.len()==3,"location2"].unique())
#print("Less than 3 words")
#print(train.loc[train["location2"].str.split().str.len()<3,"location2"].unique()[100:900])

#New colum location-targ relation between location y target to plot 
train.loc[train["location"].isnull(),"location-target"]=None
bool_nul_targ=(train["target"]==1)&(train["location"].isnull())
bool_nul_no_targ=(train["target"]==0)&(train["location"].isnull())
train.loc[bool_nul_targ,"loc-targ"]="No location,Real Desaster"
train.loc[bool_nul_no_targ,"loc-targ"]="No location,Not Real Desaster"
bool_loc_targ=(train["target"]==1)&(train["location-target"])
bool_loc_no_targ=(train["target"]==0)&(train["location-target"])
train.loc[bool_loc_targ,"loc-targ"]="Real location,Real Desaster"
train.loc[bool_loc_no_targ,"loc-targ"]="Real location,Not Real Desaster"
bool_no_loc_targ=(train["target"]==1)&( train["location-target"]==False)
bool_no_loc_no_targ=(train["target"]==0)&(train["location-target"]==False)
train.loc[bool_no_loc_targ,"loc-targ"]="Unreal location,Real Desaster"
train.loc[bool_no_loc_no_targ,"loc-targ"]="Unreal location,Not Real Desaster"


def plotKeywords(train,real):
    "real is the number: 1 or 0"
    tg = "real" if real==1 else "unreal"
    fig, ax = plt.subplots(figsize=(15,22))
    keyword_tg= train.loc[train["target"]==real,"keyword"].value_counts()
    ax = sns.barplot(x=keyword_tg.values, y=keyword_tg.index, orient='h')
    ax.set_ylabel("Keyword", fontsize=18)
    ax.set_title("Keywords of "+tg+" disaster Tweets", fontsize=25)
    ax.set_xlabel("Amount of tweets", fontsize=20)
    plt.savefig("keyword"+tg+"desastertweet.png")

def plotKeywords(train,tg):
    tg_label = "real" if tg==1 else "unreal"
    tg_color= "red" if tg==1 else "blue"
    fig, ax = plt.subplots()
    ten_keywords=train.loc[train["target"]==tg,"keyword"].value_counts().head(10)
    ten_keywords.plot(kind='bar',figsize=(10,8),rot=25, color=tg_color)
    ax.set_ylabel('Amount of tweets', fontsize=14)
    ax.set_title("Top keywords for "+tg_label+" disaster Tweets", fontsize=20)
    ax.set_xlabel('Top ten keywords', fontsize=13)
    plt.savefig("topkeyword"+tg_label+"desastertweet.png")



def plotpie(train):
    location=train["loc-targ"].value_counts()
    fig,ax=plt.subplots()
    ax=location.plot.pie(figsize=(12, 7), colors=['lightblue', 'tomato', 'royalblue',\
         'r', 'b', 'crimson'],fontsize=8.75, autopct='%.2f')
    ax.set_ylabel('')
    ax.set_title("Real, unreal or no localition, for real or not real desaster", fontsize=16)
    ax.set_xlabel('', fontsize=12)
    plt.savefig("pieLoc.png")
  

def plotboxlen(train):
    colors = ["deepskyblue", "red"]
    fig, ax = plt.subplots()
    plt.title('Amount of tweet characters that are about \n a real disaster or not'\
            ,fontsize=12,fontweight='bold')
    ax = sns.boxplot(x="target", y="len_text", 
                        data=train, palette=colors)

    ax.set_xticklabels(["Not Real Disaster","Real Disaster"])
    ax.set_ylabel('Amount of tweet characters')
    ax.set_xlabel("")
    plt.savefig('lenvsRealorNot.png')

def plotmissing(train):
    fig,ax=plt.subplots()
    datas =[["location", train["location"].isnull().sum() ],["keyword" ,\
        train["keyword"].isnull().sum() ]] 
    missing = pd.DataFrame(datas, index=['location', 'keyword'])
    ax=missing.plot(kind="bar",rot=0,figsize=(6,4), \
        color=["mediumorchid","green"],legend=False)
    ax.set_title("Amount of missing values", fontsize=14)
    ax.set_ylabel('Number')
    ax.set_xlabel("Data")
    plt.savefig('missing.png')

def amountquestion(train):
    #para acomodar
    train['question'] = train['text'].apply(lambda x: len(str(x).split("?"))-1 )
    questions=train.loc[:,["target","question"]]
    questions["value"]=[1]*len(questions["target"])
    questions=questions.groupby(["question","target"]).agg("count")
    #value of 0 amount of ? can be deleted because it is out of range
    colors = ["deepskyblue", "red"]
    fig, ax = plt.subplots()
    questions.plot(kind='bar',figsize=(10,8),rot=25, color=colors)
    plt.title('Amount of "?" in each tweet ',fontsize=12,fontweight='bold')
    #ax = sns.barplot(x=questions["value"], y=questions_get_group("question"), hue=questions_get_group("target") , orient='h')
    #ax = sns.boxplot(x="target", y="question", 
                        #data=questions, palette=colors)

    ax.set_xticklabels(["Not Real Disaster","Real Disaster"])
    ax.set_ylabel('Amount of "?" in each tweet')
    ax.set_xlabel("Aoumnt of time of that amount of ?")
    plt.savefig('questions.png')

def plotPersonalWords(train):
    pattern = r"\bi\b|\bmine\b|\bwe\b|\bmy\b|\bour\b"
    #(I , My, Our, me, we,  mine)
    bool_personal=train["text"].str.contains(pattern, flags=re.I,na=False)
    personal=train.loc[bool_personal,"target"].value_counts()
    fig,ax=plt.subplots()
    ax=personal.plot.pie(figsize=(6, 6), colors=['lightblue', 'crimson'],\
        fontsize=8.75, autopct='%.2f',labels=["Unreal Desaster","Real Desaster"])
    ax.set_title("Tweets with personals words", fontsize=12)
    ax.set_ylabel("")
    plt.savefig("piePersonal.png")

def plotNotPersonalWords(train):
    pattern = r"\bi\b|\bmine\b|\bwe\b|\bmy\b|\bour\b"
    #(I , My, Our, me, we,  mine)
    bool_personal=train["text"].str.contains(pattern, flags=re.I,na=False)
    personal=train.loc[bool_personal==False,"target"].value_counts()
    print(personal)
    fig,ax=plt.subplots()
    ax=personal.plot.pie(figsize=(6, 6), colors=['lightblue', 'crimson'],\
        fontsize=8.75, autopct='%.2f',labels=["Unreal Desaster","Real Desaster"])
    ax.set_title("Tweets without personals words", fontsize=12)
    ax.set_ylabel("")
    plt.savefig("pieNoPersonal.png")

plotNotPersonalWords(train)