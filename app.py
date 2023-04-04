import joblib
import streamlit as st
import os
from utils import *
from PIL import Image

image = Image.open('ser_pic.png')
st.image(image)
st.title("Speech Emotion Recognition Using ML")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  st.audio(uploaded_file,format = "wav")
  with open(os.path.join("/content",uploaded_file.name),"wb") as f:
    f.write(uploaded_file.getbuffer())
    
  file_path = f"/content/{uploaded_file.name}"
  X = load_data(file_path)
  text_model = joblib.load('/content/speech-emotion')
  op = text_model.predict(X)
else:
  st.stop()

if st.button('PREDICT'):
   st.subheader("Emotion: "+op[0])