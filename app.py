import streamlit as st
from PIL import Image
import pandas as pd



st.title('hello,kitty')
st.write('happy new year')
  
st.write('good morning')


# to load images;
img = Image.open("Lenna.png")
st.image(img)


# to upload dataset;
df = pd.read_csv("ks_rawdata_ng_by_SLWB000 1.csv")

st.line_chart(df['voltage'])
st.line_chart(df['e_current'])
