import streamlit as st 
import matplotlib.pyplot as plt
import os


st.subheader("Frequently Asked Questions About Streamlit")

## How to Receive User Input
st.subheader("How to Receive User Input")
name = st.text_input("Enter Your Name","Type Here")
result_name = name.title()
st.write(result_name)

## How to do Upload of Files
# Solution By Adrien Treuile @streamlit
#NB New Updates may include this feature request
def file_selector(folder_path='.'):
	filenames = os.listdir(folder_path)
	selected_filename = st.selectbox('Select a file', filenames)
	return os.path.join(folder_path, selected_filename)

	filename = file_selector()
	st.write('You selected `%s`' % filename)
