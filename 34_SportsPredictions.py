"""
Use the data set players_21 for training the model(s) and the data set players_22 for testing/evaluating the model(s):
The datasets can be found under Module 7 >> FIFA 22 Datasets
"""

#Demonstrate the data preparation & feature extraction process
#Loading the dataset I am to use
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

#Still on problem 1
#Loading dataset
df=pd.read_csv('/content/drive/My Drive/Colab Notebooks/players_21.csv')


#Data preparation
#Check for NAs and missing values, categorical values and continuous values
df.head()


df.shape

df.info() #Information about the data

#Checking for null values
null_values = df.isnull() #Returns a boolean
null_values #False - does not contain a null value

null_values_sum = null_values.sum()
null_values_sum
#The null values are in columns that I do not need

#Clean the dataset
#df = df.dropna() #Dropping all missing values
df.shape

#Separating categorical variables from numeric variables
cv = df.select_dtypes(include = ['object'])

nv = df.select_dtypes(include = ['int', 'float'])

#Filling missing values
cv_fill = cv.fillna('Nada')

#Getting rid of NaNs, putting values into it
from sklearn.impute import SimpleImputer
import numpy as np

imp_mean = SimpleImputer(strategy = 'mean') #Logic
imp_mean.fit_transform(nv)

#Encoding the categorical variables
#Categorical variables
#columns=['player_positions','potential','club_name','league_name','nationality_name','weak_foot','preferred_foot','skill_moves','international_reputation']
dummies = pd.get_dummies(df, columns=['player_positions','potential','club_name','league_name','nationality_name','weak_foot','preferred_foot','skill_moves','international_reputation'])
dummies

merged = pd.concat([nv, dummies], axis = 'columns')
merged

df.columns.tolist()


df.describe()

df = merged 

df.drop('player_url', axis= 'columns')

#feature extraction process
#Using the lab own

#Since there are a lot of columns instead of dropping I will just select my features needed

X = df[[ 'potential', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing', 'dribbling',
            'defending', 'physic', 'weak_foot', 'skill_moves', 'international_reputation']]
X

#Target or dependent variable
y = df['copy_overall']
y

#df.drop('overall', axis = 'columns')
#df.columns

df.rename(columns = {'overall': 'copy_overall'})

#Create feature subsets that show maximum correlation with the dependent variable. [5]
#Checking correlation between the columns above and the dependent variable
correlation = df.corr()['copy_overall'].abs()
#correlation['overall'].sort_values(ascending=False)

correlation #According to this overall is 100% dependent on overall

#List the max correlation with the dependent variable
#selected_max = correlation['overall'].abs().nlargest(12).index.tolist()
#selected_max

#Selecting our features needed based on the correlation



#Create and train a suitable machine learning model with cross-validation that can predict a player's rating. [5]
#Splicing the data and using the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape



X_train = X
y_train = df['copy_overall']
#Linear regression model
l=LinearRegression()
l.fit(X_train, y_train) #Trains the model using the training data
l_scores = cross_val_score(l, X_train, y_train, cv=5) #Cross validation to compute the score five times
l_scores #error says missing and NAN values from X

#Decision tree model
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train, y_train)

dt_scores = cross_val_score(l, X_train, y_train, cv=5) #Cross validation to compute the score five times
#Appears the same error here too


#Random forest model
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

rfs = cross_val_score(l, X_train, y_train, cv=5) #Cross validation to compute the score five times

#Measure the model's performance and fine-tune it as a process of optimization. [5]
#Predicting the ratings using the trained linear model
l_pred = l.predict(X_test)

#Checking the mean squared errors
l_mse = l.mean_squared_error(y_test, l_pred)

#Use the data from another season(players_22) which was not used during the training to test how good is the model. [5]
p22 = pd.read_csv('players_22.csv')
