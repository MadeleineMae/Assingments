Machine Learning Assingment 1

This project assessed and compared the effectivness of two machine learning classification models: Random Forests and XGBoost. 

The findings of the preferred model (XGBoost) was then applied to the research question of how the the characteristics of a terrorist attack affects the probability that the attack targets citizens. XGBoosts classifier model helped discover that the type of weapon and attack had more weight in increasing the probability risk of citizens being targets of a terrorist attack. These findings will be beneficial to researchers and policymakers to be able to have a clear overview of the types of terrorist attacks in order to enforce preventative measures. 

Getting Started 
Dependencies:
The libraries needed for this study include: 
import numpy as np 
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope 
from hyperopt.pyll.stochastic import sample
import pickle
from sklearn.metrics import confusion_matrix

This code was completed and executed on a MacOS. 

Installing:
The dataset comes from the Global Terrorism Database (https://www.start.umd.edu/data-tools/GTD). To download you will need to submit a request and accept the terms and conditions, then a link will be sent to your email which you will need to accept. Once you have accepted the request it will then redirect you to the download page. When the dataset is downloaded it will be in Excel format, you will need to convert it into CSV for the code to run. 

Dataset is found in .gitignore, it is extremely large so it has been compressed. 

Executing program:
Click the run command to execute the program. Then you will be able to see the print statements of each section of the code and the graphs. For the XGBoost model, change the charactersitcs of a terrorist attack by altering the 0s and 1s. This will help to change the probability and give more insight into the types of attacks which target citizens. 

Authors:
ex. Madeleine Butcher

License:
This porject is liscensed under the MIT License - see the LICENSE file for detials 


