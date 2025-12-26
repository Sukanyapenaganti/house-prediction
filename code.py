import streamlit as st
import pandas as pd
import numpy as np
import pickle
#load the model

model = pickle.load(open('house_rent_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

#page configuration

st.set_page_config(page_title="house rent prediction", page_icon=":house")
st.title("house rent prediction app")