import streamlit as st
from PIL import Image
import pandas as pd



# Sidebar contents
st.sidebar.title("DNN")
option1 = st.sidebar.selectbox("Choose a value:", 
                               ["A", "B", "C"])

slider_val = st.sidebar.slider("Select number", 0, 100, 50)

option2 = st.sidebar.radio("Choose method:", 
                           ["good", "bad", "normal"])

st.write("Selected option:", option)
st.write("Slider value:", slider_val)



# //////////////////////////////////////////////////////////

st.title('hello,kitty')
st.write('happy new year')
  
st.write('good morning')


# to load images;
img = Image.open("Lenna.png")
st.image(img)


# to upload dataset;
df = pd.read_csv("ks_rawdata_ng_by_SLWB000 1.csv")

st.subheader("voltage [V]")
st.line_chart(df['voltage'])

st.subheader("current [A]")
st.line_chart(df['e_current'])
