import streamlit as st 

def main():
	"""Deploying Streamlit App for Lip Sync"""

	st.title("Lip Sync Model")
	st.header("Wav2Lip")


	activities = ["Home","Plots"]

	choices = st.sidebar.selectbox('Select Activities',activities)

	if choices == 'Home':
		st.subheader("Home")

	elif choices == 'Plots':
		st.subheader("Visualization")




if __name__ == '__main__':
	main()
