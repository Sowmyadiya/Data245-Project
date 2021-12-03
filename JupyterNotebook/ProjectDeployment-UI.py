import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def user_input_features():
    st.title("Job Location Prediction")
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h3 style="color:white;text-align:center;">Location Prediction Model</h>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
   # basesalary= st.text_input("Basesalary","")
    year_comp = st.text_input("year_comp","")
    #bonus= st.text_input("Bonus","")
    #stocks = st.text_input("Stocks Value","")
    #company = st.selectbox("Company",levels_raw['company'].unique())
    years_of_exp = st.text_input("Years of Experience","")
    title = st.selectbox("Title",levels_raw['title'].unique())
    safety = st.selectbox("Safety",["Very Less Crime","Less Crime","Moderate Crime","High Crime","Very High Crime"])
    affordability = st.selectbox("Affordability",["Very Good","Good","Medium","Low","Very Low"])
    Economy = st.selectbox("Economy Rating",["Very Good","Good","Medium","Low","Very Low"])
    Education_health = st.selectbox("Education and Hospital Rating",["Very Good","Good","Medium","Low","Very Low"])
    Quality_of_life = st.selectbox("Quality of Life Rating",["Very Good","Good","Medium","Low","Very Low"])
    job_opportunity = st.selectbox("Job Opportunity Rating",["Very High","High","Medium","Low","Very Low"])
    climate = st.selectbox("Weather Rating",["Very Good","Good","Medium","Low","Very Low"])
    qualification = st.selectbox("Qualification",["Doctrate","Graduate","Under Graduate","High School","Other","Prefer not to say"])
    if safety == "Very Less Crime":
        safety = 0
    elif safety == "Less Crime":
        safety = 1
    elif safety == "Moderate Crime":
        safety = 2
    elif safety == "High Crime":
        safety = 3
    else:
        safety = 4 
    if affordability == "Very Good":
        affordability = 0
    elif affordability == "Good":
        affordability = 1
    elif affordability == "Medium":
        affordability = 2
    elif affordability == "Low":
        affordability = 3
    else:
        affordability = 4 
    if Economy == "Very Good":
        Economy = 0
    elif Economy == "Good":
        Economy = 1
    elif Economy == "Medium":
        Economy = 2
    elif Economy == "Low":
        Economy = 3
    else:
        Economy = 4 
    if Education_health == "Very Good":
        Education_health = 0
    elif Education_health == "Good":
        Education_health = 1
    elif Education_health == "Medium":
        Education_health = 2
    elif Education_health == "Low":
        Education_health = 3
    else:
        Education_health = 4 
    if Quality_of_life == "Very Good":
        Quality_of_life = 0
    elif Quality_of_life == "Good":
        Quality_of_life = 1
    elif Quality_of_life == "Medium":
        Quality_of_life = 2
    elif Quality_of_life == "Low":
        Quality_of_life = 3
    else:
        Quality_of_life = 4 
    if climate == "Very Good":
        climate = 0
    elif climate == "Good":
        climate = 1
    elif climate == "Medium":
        climate = 2
    elif climate == "Low":
        climate = 3
    else:
        climate = 4 
    if job_opportunity == "Very High":
        job_opportunity = 0
    elif job_opportunity == "High":
        job_opportunity = 1
    elif job_opportunity == "Medium":
        job_opportunity = 2
    elif job_opportunity == "Low":
        job_opportunity = 3
    else:
        job_opportunity = 4
    if qualification == "Doctrate":
        qualification = 0
    elif qualification == "Graduate":
        qualification = 1
    elif qualification == "Under Graduate":
        qualification = 2
    elif qualification == "High School":
        qualification = 3
    elif qualification == "other":
        qualification = 4
    else:
        qualification = -1


    # data = {'basesalary': basesalary,
    #             'bonus': bonus,
    #             'stockgrantvalue': stocks,
    #             'company':company,
    #             'yearsofexperience':years_of_exp,
    #             'title':title,
    #             'tag':tag,
    #             'safety':safety,
    #             'affordability':affordability,
    #             'economy':Economy,
    #             'education and health':Education_health,
    #             'quality of life':Quality_of_life,
    #             'emp_population_ratio':job_opportunity,
    #             'Climate':climate,
    #             'Education':qualification
                
    #             }
    data = {'totalyearlycompensation': year_comp,
                'yearsofexperience':years_of_exp,
                'title':title,
                'safety':safety,
                'affordability':affordability,
                'economy':Economy,
                'education and health':Education_health,
                'quality of life':Quality_of_life,
                'job_opportunity_rank':job_opportunity,
                'Climate':climate,
                'Education':qualification
                
                }

                
    features = pd.DataFrame(data, index=[0])
    return features


levels_raw = pd.read_csv('https://raw.githubusercontent.com/Sowmyadiya/Data245-Project/master/Datasets/proj_data245_testing.csv')
input_df = user_input_features()



levels = levels_raw.drop(columns=['state'])
df = pd.concat([input_df,levels],axis=0)


load_clf = pickle.load(open("AdaBoost_data245.pkl", 'rb'))

# Apply model to make predictions
if st.button("Predict"):
    encode = ['title']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df.drop(columns=['Unnamed: 0'])
    print(df)
    print('****')
    # pca = PCA(n_components=50, random_state=11)  # reduce to two components
    # pca.fit(df)
    df = df[:1]
    
    print(df)

    # data = pca.transform(df)
    #print(data)
    prediction = load_clf.predict(df)
    #prediction_proba = load_clf.predict_proba(df)
    st.write(prediction)


