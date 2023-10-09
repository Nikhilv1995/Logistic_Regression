# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:12:02 2023

@author: nikhilve
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data=pd.read_csv("Social_Network_Ads.csv")


#finding unique values.
data['User ID'].unique()


#finding count of each value.
data['User ID'].value_counts()



#selecting features and result.   OR Vector of DV(Dependent Variables) y, and Matrix of IV(Independent Variables) x
x=data.iloc[:,[2,3]].values


y=data.iloc[:,4].values


#Feature Scaling- Here we are scaling all the data into the same scale.
#can be done before train test split, then it would be easy.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)




#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

# DATA PRE-PROCESSING IS DONE TILL HERE, NOW WE WILL IMPLEMENT THE MODEL.

#======================================================================================================

# Implementing Classification using Logistic Regression algo

from sklearn.linear_model import LogisticRegression
#creating an object named classifier of logistic regression model.
classifier= LogisticRegression()

#Training the model
classifier.fit(x_train,y_train)

#Using trained model for doing predictions
y_pred = classifier.predict(x_test)

y_pred1 = classifier.predict_proba(x_test)



#Checking the accuracy of our model using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)



print("Values of X1 and X2 respectively :",classifier.coef_)
print("Value of W0 :",classifier.intercept_)


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Accuracy:", accuracy)







