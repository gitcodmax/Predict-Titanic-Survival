# Predict Survival on the Titanic

## 📌 Project Overview

The Predict Titanic Survival project is a machine learning project based on Kaggle’s [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition. 
The objective is to build a classification model that predicts whether a passenger survived the Titanic disaster based on passenger and voyage information.

The project covers the complete machine learning workflow, including data exploration, data cleaning, feature engineering, model training, 
cross-validation, hyperparameter tuning, and model evaluation. Different classification algorithms are explored and compared to identify a model that performs well on unseen data. 

## 🎯 Problem Statement

The objective is to develop a binary classification model that predicts whether a passenger survived the Titanic disaster based on their available characteristics. 
The model should learn patterns from historical passenger data and accurately classify unseen passengers as survived (1) or did not survive (0).

## 📊 Dataset

To build the best model, Kaggle provides the following features on the Titanic.

| Column | Definition |
| --- | --- |
| PassengerId | Id of the passengers |
| Survived | Did the passenger survive? 0(No) 1(Yes) |
| Pclass | Ticket class |
| Name | Passenger name |
| Sex | Gender |
| Age | Age |
| Sibsp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Point of embarkation |

Download the dataset from [here](https://www.kaggle.com/competitions/titanic/data)

## 📁 File Descriptions

| File/Folder | Description |
| --- | --- |
| models/ | Notebooks based on different algorithms |
| proc_data/ | csv files on processed data |
| submissions/ | csv files containing predictions derived from different models |
| custom_func.py | Python file containing user defined functions used in the notebooks |
| titanic.ipynb | Contains the steps for processing the data |
| gender_submission.csv, test.csv, train.csv | Sample submission file, test and training data respectively |

## 🔎 Exploratory Data Analysis

The datasets used have the following structure:  
- 891 rows and 12 columns for training
- 418 rows and 11 columns for testing
- Youngest and oldest passengers are 0 and 80 years respectively
- Most tickets purchased belong to second class
- Most records on the training set are of passengers that did not survive
- Most females survived compared to men

## 🛠️ Data Preprocessing

The following activities were carried out to ensure that the models was working clean data:
- Drop the PassengerId, Ticket and Cabin columns
- Fill null values in age with the average age
- Drop 2 records missing a values for Embarked column
- Fill null values in fare with average fare

## ⚙️ Feature Engineering

FE was carried out in the ```models/mod_updates.ipynb``` file. It includes:
- Log transformed Fare to work with normally distributed data
- Create FamilySize feature by adding no. of siblings(Sibsp) and no. of parents(Par)
- Create IsAlone feature for passengers without any family member alone
- Categorize the ages, Child, Teen, Adult, Middle-Age and Senior
- Create a titles feature from the Name column, Mr., Miss., Mrs., Master. and Other

## 🤖 Models

The table below shows the validation and Kaggle scores achieved by different models.

| Model | Validation Score | Kaggle Score |
|--- | --- | --- |
| Logistic Regression | 0.802 | 0.765 |
| Decision Tree | 0.798 | 0.760 |
| Random Forest | 0.833 | 0.777 |
| XGBClassifier | 0.830 | 0.765 |

The RANDOM FOREST model, with the highest score, was able to generalize better compared to other algorithms.

## 🚀 How to Run the Project

1. Clone the repository
```
https://github.com/gitcodmax/Predict-Titanic-Survival.git
cd "project_folder"
```

2. Create a virtual environment
```
python -m venv venv
```

  Activate it on Windows:
  
```
venv\Scripts\activate
```
3. Install dependencies
```
 pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

4. Obtain the dataset

Download the dataset from the Kaggle competition:  
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)

5. Run the notebooks  
Open the notebooks using Jupyter Notebook or VS Code and execute them.

```                                    MMAX CODES                                                   ```
