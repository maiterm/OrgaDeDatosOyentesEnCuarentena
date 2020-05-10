import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sb = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


print("sample_submission")
print(sb.tail())

print("test")
print(test.tail())
print("Test amount of rows",len(test))
print("train")
print(train.tail())
print("Train amount of rows",len(train))

print("test locations")
print(test["location"].unique())
print("len test unique locations",len(test["location"].unique()))
print("len test location == Nan",len(test[test["location"].isnull()]))


print("train locations")
print(train["location"].unique())
print("len train unique locations",len(train["location"].unique()))
print("len train location == Nan",len(train[train["location"].isnull()]))

print("test keywords")
print(test["keyword"].unique())
print("len test unique keywords",len(test["keyword"].unique()))
print("train keywords")
print(train["keyword"].unique())
print("len train unique keywords",len(train["keyword"].unique()))

