import streamlit as st
st.markdown("# myapp")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#loading data file
s = pd.read_csv("C:/Users/19496/Desktop/Module 2/social_media_usage.csv")


#import, clean data

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)
ss=pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "marital":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])

})
ss=ss.dropna()

#set y and x variable 
y=ss["sm_li"]
x=ss[["income","education","parent","marital","female","age"]]

#80/20 split of train and test data 
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=123)

#fit data 
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train,y_train)


#create app 
st.markdown("# Linkedin User Prediction")


#age slider 
age = st.slider("Select Your Age", min_value=18, max_value=98)


#radiobutton for gender, marital status, parent 
genderradio = st.radio("Select Your Gender", ("Male", "Female"))
if genderradio == "Male":
    gender = 0
else:
    gender = 1 
maritalradio = st.radio("Select Your Martial Status", ("Not Married", "Yes, Married"))
if maritalradio == "Yes, Married":
    marital = 1
else:
    marital = 0 
parentradio = st.radio("Select Your Parental Status", ("No Children", "Yes, Have Children"))
if parentradio == "Yes, Have Children":
    parent = 1
else:
    parent = 0 


#education selection box 
education = st.selectbox(label="Select Your Highest Level of Education",
    options=("1 Less Than High School","2 High School Incomplete","3 High School Graduate",
    "4 Some College","5 Associate Degree - 2 year","6 College or University Degree - 4 Year","7 Some Postgraduate Schooling",
    "8 Postgraduate or Professional Degree"))


#income selection box 
income = st.selectbox(label="Select Your Yearly Income Range",
    options=("1 - Less than $10k","2 - $10k - 20k","3 - $20k - 30k", "4 - $30k - 40k","5 - $40k - 50k",
    "6 - $50k - 75k","7 - $75k - 100k", "8 - $100k - $150k", "9 - Over $150k"))

#create list with inputs from answers above 
answer = [int(income[0]),int(education[0]),parent,marital,gender,age]

#run list in lr prediction 
predicted_class = lr.predict([answer])
probs = lr.predict_proba([answer])

#print results 
if predicted_class == 1:
    st.write("You are predicted to be a LinkedIn user.")
else:
    st.write("You are not predicted to be a LinkedIn user.")
st.write("The probability you are a LinkedIn user is:", {probs[0][1]})







