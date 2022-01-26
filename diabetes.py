import streamlit as st

import pandas as pd

from sklearn.tree import DecisionTreeClassifier


def main():
    try:
        st.title('Diabetes Predictor')

        df = pd.read_csv('diabetes.csv')

        X = df.drop(['Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction', 'BMI', 'Outcome'], axis=1)

        Y = df['Outcome']

        pre = DecisionTreeClassifier()

        pre.fit(X, Y)

        Glucose = st.text_input('Enter Blood Glucose Level: ')

        Insulin = st.text_input('Enter Insulin Level: ')

        Age = st.text_input('Enter Your Age: ')

        res = pre.predict([[Glucose, Insulin, Age]])

        st.title(res)

    except ValueError:
        st.write('Please Enter Something Only Numbers')

main()
