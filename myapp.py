import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

def app():
    img = Image.open(r'img.jpeg')
    img = img.resize((200,200))
    st.image(img,caption='Diabetes Image',width=200)

    st.title('Diabetes Disease Prediction')

    st.sidebar.title('Input Features')




    if __name__=="__main__":
        app()