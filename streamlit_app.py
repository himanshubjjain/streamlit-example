import streamlit as st 

def main():
	"""Deploying Streamlit App for Lip Sync"""

	st.title("Lip Sync Model")
	st.header("Wav2Lip")

	st.title("Updated Version")


	activities = ["EDA","Plots"]

	choices = st.sidebar.selectbox('Select Activities',activities)

	if choices == 'EDA':
		st.subheader("EDA")

	elif choices == 'Plots':
		st.subheader("Visualization")




if __name__ == '__main__':
	main()
