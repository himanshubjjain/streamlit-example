import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs

from PIL import Image 

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 



def main():
	st.title("Lip Sync")

	menu = ["Image","Audio","Output"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Image":
		st.subheader("Image")
		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
		if image_file is not None:
		
			# To See Details
			# st.write(type(image_file))
			# st.write(dir(image_file))
			file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
			st.write(file_details)

			img = load_image(image_file)
			st.image(img)
		


	elif choice == "Audio":
		st.subheader("Audio")
		audio_file = st.file_uploader("Upload Audio",type=['wav'])
		if st.button("Process"):
			if audio_file is not None:
				file_details = {"Filename":audio_file.name,"FileType":audio_file.type,"FileSize":audio_file.size}
				st.write(file_details)

				audio_bytes = audio_file.read(audio_file)
				st.audio(audio_bytes,format='audio/wav')
				
if __name__ == '__main__':
	main()					
