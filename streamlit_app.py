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
	st.title("Lip Sync")

name = st.text_input("Input the message","Type Here")
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
menu = ["Home","Custom_Settings","About"]
choice = st.sidebar.selectbox("Menu",menu)
