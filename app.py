import streamlit as st
# import pandas as pd
# import numpy as np
from joblib import load
import pickle
import helper

model = pickle.load(open("model.pkl", "rb"))
# model = load('model.joblib')

st.header('Duplicate Question Pairs Identifier')

q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

if st.button('Find'):
    query = helper.input_array(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header("DUPLICATE")
    else:
        st.header("NOT DUPLICATE")



