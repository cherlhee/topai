import streamlit as st
from PIL import Image
import pandas as pd

# ---------------------------
# MAKE FULL WIDTH
# ---------------------------
st.set_page_config(layout="wide")


# //////////////////////////////////////////////////////////
# Sidebar contents
st.sidebar.title("DNN")
option1 = st.sidebar.selectbox("Choose a value:", 
                               ["A", "B", "C"])

slider_val = st.sidebar.slider("Select number", 0, 100, 50)

option2 = st.sidebar.radio("Choose condition:", 
                           ["good", "bad", "normal"])

# //////////////////////////////////////////////////////////
# to create tabs;
tab1, tab2, tab3 = st.tabs(['TAB-1', 'TAB-2', 'TAB-3'])

with tab1:
  st.header('tab-one')

with tab2:
  st.header('tab-two')
  st.image('co2weld.jpg')
with tab3:
  st.header('tab-three')
  st.image('Lenna.png')
    
# //////////////////////////////////////////////////////////

st.title('DNN Dashboard')
st.write('happy new year')
st.write('good morning')


st.write("Selected option:", option1)
st.write("Selected condition:", option2)
st.write("Slider value:", slider_val)


# to load images;
# img = Image.open("Lenna.png")
img = Image.open("co2weld.jpg")
st.image(img)


# to upload dataset;
df = pd.read_csv("ks_rawdata_ng_by_SLWB000 1.csv")

st.subheader("voltage [V]")
st.line_chart(df['voltage'])

st.subheader("current [A]")
st.line_chart(df['e_current'])
