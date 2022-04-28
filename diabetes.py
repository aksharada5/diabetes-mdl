import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pickle

df1 = pd.read_csv('diabetes.csv')
df2 = pd.read_csv('new_diabetes.csv')
df=df1.append(df2)
df.drop_duplicates()
X = df.drop("Outcome", axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,random_state=50)

# Creating model object
model_dt = DecisionTreeClassifier( max_depth=16, random_state=40)
# Training Model
model_dt.fit(X_train,y_train)

# Making Prediction
pred_dt = model_dt.predict(X_test)
# Calculating Accuracy Score
dt = accuracy_score(y_test, pred_dt)
print(dt)

# Calculating Precision Score
dt = precision_score(y_test, pred_dt)
print(dt)

# Calculating Recall Score
dt = recall_score(y_test, pred_dt)
print(dt)

# Calculating F1 Score
dt = f1_score(y_test, pred_dt)
print(dt)

# confusion Maxtrix
cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot = True, fmt=  '0.2%', cmap = 'Reds')

filed='diabetes_model'
pickle.dump(model_dt,open(filed,'wb'))
