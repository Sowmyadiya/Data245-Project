## Data245-Project
Job Location Suggesstion Using Machine Learning
People are currently considering where to settle based on work availability and living conditions. Researching job availability, location, payscale and other factors is time consuming and also there is no mechanism for determining a person’s ideal job and settling location.The goal of the project is to build a location recommendation system for job-searchers to help them settle in a tranquil environment that suits their preferences. Many websites such as glass door , levels fyi  predict salaries based on various attributes such as location , job roles ,years of experience and so on. But not the location based on the individual's preference. 


## Technologies Used:

1) The datasets collected from various sources are merged and explored using **pandas, numpy, seaborn, matplotlib and KNNImputer libraries**.
2) **Equal Frequency and Equal Width binning** is performed to group related values in bins to reduce the number of distinct values.
3) **One-Hot encoding** to convert the categorical variables into a form that can be provided to ML algorithm for better predictions
4) This is a **supervised multiclass classification** problem and hence the below models are proposed:
  Models:
    1) Decision Tree
    2) K-Nearest Neighbor
    3) AdaBoost
    4) XGBoost
5) The models are evaluated using the following evaluation metrices:
   1) Precision
   2) Recall
   3) F1 Score
   4) Confusion Matrix
   5) ROC-AUC curves
   6) Learning Curves
   
6) Interactive UI Build using the **Streamlit** (An open-source Python library) 

## Steps Involved:

1) The datasets are collected from various sources:
  Levels_FYI            - Contains information regarding job postings (salary, 	location, 	title, etc.)
  Livability_Score      - States’ rankings based on affordability, economy, cost of 	living, etc.
  Weather_Data          - Average daily temperature and rank based on other factors
  Tech_Employment_Count - Contains each state’s tech workforce amount
  
2) Datasets are merged using the state column.
3) Data cleaning, binning, and one hot encoding is done in the pre- processing stage.
4) The target feature is identified as the states in the US.
5) The dataset is split into training, validation and test sets.
6) The data is given to the Machine Learning models.
7) Hypertuning is performed to improve the models performance and to avoid overfitting.
8) The model is eveluated based on evaluation metrices.
9) Building interactive UI to suggest job location based on users preference.

## Summary

  <img width="732" alt="Screen Shot 2022-02-11 at 11 27 21 PM" src="https://user-images.githubusercontent.com/49642360/153701734-d8a53f3d-89ca-4d7e-9238-a88e73daab16.png">
