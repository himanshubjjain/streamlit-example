# Core Pkgs
import streamlit as st
import streamlit.components.v1 as stc 
import os 

st.set_page_config(
     page_title='Streamlit cheat sheet',
     layout="wide",
     initial_sidebar_state="expanded",


 name = st.text_input("Input the message","Type Here")
 result_name = name.title()
 st.write(result_name)
)
