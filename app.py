import streamlit as st
import plotly_express as px
import numpy as np
import pickle

from predict import predict_flower

st.title("My Iris Predictor")
st.header("Lets predict Iris species")

#use cache to speed loading
@st.cache()
def load_data():
    return px.data.iris()

df_iris = load_data()

hist_sl = px.histogram(df_iris, x = 'sepal_length')
hist_sl

show_df = st.checkbox("Do you want to see the data?")
if show_df:
    df_iris

sl = st.number_input("Sepal Length (cm)", 0, 100)
sw = st.number_input("Sepal Width (cm)", 0, 100)
pl = st.number_input("Petal Length (cm)", 0, 100)
pw = st.number_input("Petal Width (cm)", 0, 100)

# TFAE
# st.write(sl)
# sl

user_input = np.array([[sl, sw, pl, pw]])
user_input

with st.spinner("Making prediction"):
    with open("saved-iris-model.pkl", "rb") as f:
        classifier = pickle.load(f)

    prediction = predict_flower(classifier, user_input)

st.header(f"Ther model predicsts: {prediction[0]}")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")
with col2:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")
with col3:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

st.sidebar.button("This button doesn't do anything")
