## Job Location Suggestion Using Machine Learning
Job Location Suggesstion Using Machine Learning
People are currently considering where to settle based on work availability and living conditions. Researching job availability, location, payscale and other factors is time consuming and also there is no mechanism for determining a person’s ideal job and settling location.The goal of the project is to build a location recommendation system for job-searchers to help them settle in a tranquil environment that suits their preferences. Many websites such as glass door , levels fyi  predict salaries based on various attributes such as location , job roles ,years of experience and so on. But not the location based on the individual's preference. 


## Python libraries, and models used

- The datasets collected from various sources are merged and explored using **pandas, numpy, seaborn, matplotlib, KNNImputer libraries, Streamlit(for UI)**.
- **Models Exprimented**
    - Decision Tree
    - K-Nearest Neighbor
    - AdaBoost
    - XGBoost
- AdaBoost achieved accuracy rate of 99.89%
  AdaBoost has highest True positive rate compare to all other models.

## Steps Involved

- The datasets are collected from various sources:
  Levels_FYI            - Contains information regarding job postings (salary, 	location, 	title, etc.)
  Livability_Score      - States’ rankings based on affordability, economy, cost of 	living, etc.
  Weather_Data          - Average daily temperature and rank based on other factors
  Tech_Employment_Count - Contains each state’s tech workforce amount
- Datasets are merged using the state column.
- **Equal Frequency and Equal Width binning** is performed to group related values in bins to reduce the number of distinct values.
- **One-Hot encoding** to convert the categorical variables into a form that can be provided to ML algorithm for better predictions.
- The target feature is identified as the states in the US.
- The dataset is split into training, validation and test sets.
- The data is given to the Machine Learning models.
- Hypertuning is performed to improve the models performance and to avoid overfitting.
- The model is eveluated based on evaluation metrices.
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC-AUC curves
  - Learning Curves
- Interactive UI to suggest job location based on users preferences.

## Summary
- Below image shows the final output where the users can input their job search prefrences, Based on the users input we leverage machine learning techniques to display the desired job location in the UnitedStates. 

  <img width="732" alt="Screen Shot 2022-02-11 at 11 27 21 PM" src="https://user-images.githubusercontent.com/49642360/153701734-d8a53f3d-89ca-4d7e-9238-a88e73daab16.png">
