# Core Pkgs
import streamlit as st
import streamlit.components.v1 as stc 
import os 

# HTML

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px;font-size:{}px">
    <h1 style="color:white;text-align:center;">Streamlit is Awesome </h1>
    <h1 style="color:white;text-align:center;">Session State is Here!! </h1>
    </div>
    """

def main():
	"""Deploying Streamlit App for Lip Sync"""

	st.title("Lip Sync Model")
	st.header("Wav2Lip")


name = st.text_input("Input the message","Type Here")
result_name = name.title()
st.write(result_name)

