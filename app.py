import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my Machine Learning Dashboard

This dashboard was created by: [@yourname](https://www.linkedin.com/in/shellatheresyapandiangan/)
""")

def iris():
    st.write("""
    ## This app predicts the **Iris Species**
    Data obtained from the [Iris dataset](https://www.kaggle.com/uciml/iris) by UCIML.
    """)
    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.header('Manual Input')
        def user_input_features():
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.3, 10.0, 6.5)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0, 5.0, 3.3)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0, 9.0, 4.5)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1, 5.0, 1.4)
            data = {
                'SepalLengthCm': SepalLengthCm,
                'SepalWidthCm': SepalWidthCm,
                'PetalLengthCm': PetalLengthCm,
                'PetalWidthCm': PetalWidthCm
            }
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    img = Image.open("Iris.jpg")
    st.image(img, width=500)

    if st.sidebar.button('Predict!'):
        with open("generate_iris.pkl", 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict(input_df)
        result = [
            'Iris-setosa' if prediction[0] == 0 else
            'Iris-versicolor' if prediction[0] == 1 else
            'Iris-virginica'
        ]
        st.subheader('Prediction:')
        with st.spinner('Predicting...'):
            time.sleep(2)
            st.success(f"The prediction is: **{result[0]}**")

# Call the Iris prediction function
iris()
