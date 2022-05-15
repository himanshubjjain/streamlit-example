import streamlit as st
import streamlit.components.v1 as stc
import torch

from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import dlib, json, h5py, subprocess
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.models import Model


# import json, subprocess, random, string
from glob import glob

import platform
# import torch
# st.write(sys.path)
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()


tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

sys.path.insert(1,'/content/LipGAN/')      

import generator
# from generator import create_model_residual as create_model


# parser = argparse.ArgumentParser(description='Code to generate talking face using LipGAN')

# parser.add_argument('--n_gpu', default=1)

# args = parser.parse_args()
# im_size = 96


checkpoint_path_wav = '/content/Wav2Lip/checkpoints/wav2lip_gan.pth'

# parser.add_argument('--audio', type=str, 
# 					help='Filepath of video/audio file to use as raw audio source', required=True)

outfile = 'results/result_voice.mp4'

pads_wav =[0, 10, 0, 0]

face_det_batch_size_wav =16

wav2lip_batch_size =128

resize_factor =1

crop =[0, -1, 0, -1]

box =[-1, -1, -1, -1]

rotate = False

nosmooth =False 




# import audio
checkpoint_path_lipgan ='/content/LipGAN/logs/lipgan_residual_mel.h5'
model = 'residual'
face_det_checkpoint = '/content/LipGAN/logs/mmod_human_face_detector.dat'
static = False
fps = 25.
max_sec = 240.
pads = [0, 0, 0, 0]
face_det_batch_size = 64
lipgan_batch_size =256
n_gpu = 1
im_size = 96
face = "nan"


# parser.add_argument('--results_dir', type=str, help='Folder to save all results into', default='results/')



# text = "One victory does not make us conquerors. Did we free my father, did we rescue " \
#        "my sisters from the Queen? Did we free the North from those who want us on our knees?"
# File Processing Pkgs

from PIL import Image


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def rect_to_bb(d):
  x = d.rect.left()
  y = d.rect.top()
  w = d.rect.right() - x
  h = d.rect.bottom() - y
  return (x, y, w, h)

def calcMaxArea(rects):
  max_cords = (-1,-1,-1,-1)
  max_area = 0
  max_rect = None
  for i in range(len(rects)):
    cur_rect = rects[i]
    (x,y,w,h) = rect_to_bb(cur_rect)
    if w*h > max_area:
      max_area = w*h
      max_cords = (x,y,w,h)
      max_rect = cur_rect	
  return max_cords, max_rect
  
def face_detect(images):
  detector = dlib.cnn_face_detection_model_v1(face_det_checkpoint)

  batch_size = face_det_batch_size

  predictions = []
  for i in tqdm(range(0, len(images), batch_size)):
    predictions.extend(detector(images[i:i + batch_size]))
  
  results = []
  pady1, pady2, padx1, padx2 = pads
  for rects, image in zip(predictions, images):
    (x, y, w, h), max_rect = calcMaxArea(rects)
    if x == -1:
      results.append([None, (-1,-1,-1,-1), False])
      continue
    y1 = max(0, y + pady1)
    y2 = min(image.shape[0], y + h + pady2)
    x1 = max(0, x + padx1)
    x2 = min(image.shape[1], x + w + padx2)
    face = image[y1:y2, x1:x2, ::-1] # RGB ---> BGR

    results.append([face, (y1, y2, x1, x2), True])
  
  del detector # make sure to clear GPU memory for LipGAN inference
  return results 

def datagen(frames, mels):
  img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

  # if not static:
  face_det_results = face_detect([f[...,::-1] for f in frames]) # BGR2RGB for CNN face detection
  # else:
  #   face_det_results = face_detect([frames[0][...,::-1]])

  for i, m in enumerate(mels):
    idx = i%len(frames)
    frame_to_save = frames[idx].copy()
    face, coords, valid_frame = face_det_results[idx].copy()
    if not valid_frame:
      print ("Face not detected, skipping frame {}".format(i))
      continue

    face = cv2.resize(face, (im_size, im_size))

    img_batch.append(face)
    mel_batch.append(m)
    frame_batch.append(frame_to_save)
    coords_batch.append(coords)

    if len(img_batch) >= lipgan_batch_size:
      img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

      img_masked = img_batch.copy()
      img_masked[:, im_size//2:] = 0

      img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
      mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

      yield img_batch, mel_batch, frame_batch, coords_batch
      img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

  if len(img_batch) > 0:
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, im_size//2:] = 0

    img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    yield img_batch, mel_batch, frame_batch, coords_batch

####Wav2LIP Functions

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect_wav(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = face_det_batch_size_wav
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = pads_wav
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen_wav(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if box[0] == -1:
		if not static:
			face_det_results = face_detect_wav(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect_wav([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (img_size, img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' 
# if torch.cuda.is_available() else 'cpu'
# print('Using {} for inference.'.format(device))

def _load(checkpoint_path_wav):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path_wav)
	else:
		checkpoint = torch.load(checkpoint_path_wav,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()







def main():
    st.title("Lip Sync")

    # menu = ["Image","text"]
    # choice = st.sidebar.selectbox("Menu", menu)


    # if choice == "Image":
    #     st.subheader("Image")
    image_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
    img_0 = Image.open(image_file)
    img0 = img_0.save("img.jpg")
    # if image_file is not None:
    #     # To See Details
    #     # st.write(type(image_file))
    #     # st.write(dir(image_file))
    #     # file_details = {"Filename": image_file.name, "FileType": image_file.type, "FileSize": image_file.size}
    #     # st.write(file_details)
    #     img_path = image_file.name
    #     img = load_image(image_file)
    #     face = img_path
        # st.write(pathh)

        # elif choice == "text":
    text = st.sidebar.text_input("Enter the text you want")

    with st.form(key ='Form1'):
            with st.sidebar:
                submit = st.form_submit_button(label = 'Submit 🔎')

    if submit:
        # face = image_file.name
        # st.write(face)
        import torch
        st.title(text)
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        sequences, lengths = utils.prepare_input_sequence([text])

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)
            audio = waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050

        from scipy.io.wavfile import write
        write("audio.wav", rate, audio_numpy)
        
        

        face = '/content/img.jpg'


        audio_path = '/content/audio.wav'

        results_dir = "/content"
        fps = 25.
        fps = fps
        mel_step_size = 27
        mel_idx_multiplier = 80./fps

        # if model == 'residual':
        from generator import create_model_residual as create_model
        # else:
        #   from generator import create_model as create_model
        if face.split('.')[1] in ['jpg', 'png', 'jpeg']:
          # st.write("HIIII")
          full_frames = [cv2.imread(face)]
        else:
          video_stream = cv2.VideoCapture(face)
          
          full_frames = []

          while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
              video_stream.release()
              break
            full_frames.append(frame)
            if len(full_frames) % 2000 == 0: print(len(full_frames))

            if len(full_frames) * (1./fps) >= max_sec: break

          print ("Number of frames available for inference: "+str(len(full_frames)))

        import audio
        from audio import load_wav
        wav = audio.load_wav(audio_path, 16000)

        mel = audio.melspectrogram(wav)
        st.write(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
          raise ValueError('Mel contains nan!')

        mel_chunks = []
        i = 0
        while 1:
          start_idx = int(i * mel_idx_multiplier)
          if start_idx + mel_step_size > len(mel[0]):
            break
          mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
          i += 1

        st.write("Length of mel chunks: {}".format(len(mel_chunks)))
        # st.write("batch: {}".format(lipgan_batch_size))

        batch_size = lipgan_batch_size
        gen = datagen(full_frames.copy(), mel_chunks)
        st.write(full_frames)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
          
          if i == 0:
            # st.write(args)
            model = create_model(0, mel_step_size)
            st.write("Model Created")

            model.load_weights(checkpoint_path_lipgan)
            st.write("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(path.join(results_dir, 'result.avi'), 
                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

          pred = model.predict([img_batch, mel_batch])
          pred = pred * 255
          
          for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p, (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

        out.release()

        os.system('ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, path.join(results_dir, 'result.avi'), 
                                  path.join(results_dir, 'result_voice.avi')))
        os.system('ffmpeg -i {} {} -y'.format('/content/result_voice.avi', '/content/output.mp4'))
        # subprocess.call(command, shell=True)
        st.write("LIPGAN Done")


        ## starting Wave2Lip inference



        
        sys.path.insert(0,'/content/Wav2Lip/') 

        import audio 
        import torch, face_detection
        from models import Wav2Lip

        face  = "/content/result_voice.avi"
        audio = '/content/audio.wav'


        if not os.path.isfile(face):
          raise ValueError('--face argument must be a valid path to video/image file')

        elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
          full_frames = [cv2.imread(face)]
          fps = fps

        else:
          video_stream = cv2.VideoCapture(face)
          fps = video_stream.get(cv2.CAP_PROP_FPS)

          print('Reading video frames...')

          full_frames = []
          while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
              video_stream.release()
              break
            if resize_factor > 1:
              frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

            if rotate:
              frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        print ("Number of frames available for inference: "+str(len(full_frames)))

        if not audio.endswith('.wav'):
          print('Extracting raw audio...')
          command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio, 'temp/temp.wav')

          subprocess.call(command, shell=True)
          audio = 'temp/temp.wav'

        wav = audio.load_wav(audio, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
          raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
          start_idx = int(i * mel_idx_multiplier)
          if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
          mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
          i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = wav2lip_batch_size
        gen = datagen_wav(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
          if i == 0:
            model = load_model(checkpoint_path_wav)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

          img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
          mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

          with torch.no_grad():
            pred = model(mel_batch, img_batch)

          pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
          
          for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, 'temp/result.avi', outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
