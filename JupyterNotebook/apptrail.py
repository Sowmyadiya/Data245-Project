


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#app=Flask(__name__)
#Swagger(app)

data = open("data.pkl","rb")
df=pickle.load(data)
RF = open("RandomForest.pkl","rb")
BC=open("BaggingClassifier.pkl","rb")
RandomForest=pickle.load(RF)
BaggingClass=pickle.load(BC)

#@app.route('/')
def welcome():
    return "Welcome All"
def load_data():
    data = pd.read_csv('data.csv')
    return data

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(totalyearlycompensation,safety,classifier):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[totalyearlycompensation,safety]])
    print(prediction)
    return prediction



def main():
    df=load_data()
    final_data_X=df[["totalyearlycompensation","safety"]]
    target=df["name"]
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(final_data_X,target,test_size=0.30,random_state=42)

    st.title("Location Prediction")
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">state prediction model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    totalyearlycompensation = st.text_input("totalyearlycompensation","Type Here")
    safety = st.text_input("safety","Type Here")
    

    model_choice = st.selectbox("Select Model",["RandomForest","BaggingClass"])
    result=""
    if st.button("Predict"):
        if model_choice == "RandomForest":
            rnd_clf = RandomForestClassifier(n_estimators=16, max_leaf_nodes=60, random_state=70,criterion='gini')
            rnd_clf.fit(Xtrain, Ytrain)
            y_pred_rf= rnd_clf.predict(Xtest)
            result=predict_note_authentication(totalyearlycompensation,safety,RandomForest)
            acc=accuracy_score(Ytest, y_pred_rf)
            acc1=int(acc * 100)
            st.success('The output is {}'.format(result))
            st.success('The accuracy is {}%'.format(acc1))
        if model_choice == "BaggingClass":
            bag_clf = BaggingClassifier(n_estimators=4,max_samples=5)
            bag_clf.fit(Xtrain, Ytrain)
            y_pred = bag_clf.predict(Xtest)
            result=predict_note_authentication(totalyearlycompensation,safety,BaggingClass)
            acc=accuracy_score(Ytest, y_pred)
            acc1=int(acc * 100)
            st.success('The output is {}'.format(result))
            st.success('The accuracy is {}%'.format(acc1))
           



if __name__=='__main__':
    main()
    



