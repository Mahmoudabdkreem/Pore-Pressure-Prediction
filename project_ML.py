import pickle
import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np

# upload data
with open('scaler.pkl','rb')as scaler_file:
  loaded_scaler=pickle.load(scaler_file)
with open('random_forest_model.pkl','rb')as model_file:
    loaded_model=pickle.load(model_file)

st.title('Pore Pressure Prediction')

st.sidebar.header('features selection')

#'TVD', 'DT', 'GR', 'RES', 'NEUT', 'DEN', 'PP'
TVD=st.sidebar.text_input('TVD')
DT=st.sidebar.text_input('DT')
GR=st.sidebar.text_input('GR')
RES=st.sidebar.text_input('RES')
NEUT=st.sidebar.text_input('NEUT')
DEN=st.sidebar.text_input('DEN')

df=pd.DataFrame({
    'TVD':[TVD],
    'DT':[DT],
    'GR':[GR],
    'RES':[RES],
    'NEUT':[NEUT],
    'DEN':[DEN]
},index=[0])
predict=st.sidebar.button('predict')
if predict:
 x_new_scaled=loaded_scaler.transform(df)
 predictions=loaded_model.predict(x_new_scaled)
 st.write(f'Pore Pressure ={predictions}')