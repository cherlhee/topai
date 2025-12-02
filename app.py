import streamlit as st
from PIL import Image



st.title('hello,kitty')
st.write('happy new year')
  
st.write('good morning')

img = Image.open(img)
st.image(img)
