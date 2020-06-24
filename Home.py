import streamlit as st,time
import numpy as np
import pandas as pd 
from PIL import Image
from io import BytesIO
import requests
import os
import matplotlib.image as mpimg
from fastai.vision import open_image,load_learner,image,torch


st.title('Tiger VS Cheetah')

st.write('## This application classifies the given image as Tiger, White tiger or Cheetah')



def predict(img):
	with st.spinner('Please wait..'):
		time.sleep(3)

	model = load_learner('.')
	pred_class,pred_idx,outputs = model.predict(img)
	pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
	st.write('## This is ',str(pred_class).capitalize(),' with probablity of ',pred_prob,'%')



options = st.radio('',('Select Image','Enter Image URL','Upload an image'))
#Select image
if options =='Select Image':
	static = os.listdir('static')
	test_image = st.selectbox('Please select a image:',static)
	file_path = 'static/' + test_image
	pil_img =Image.open(file_path)

	#read the image
	img = pil_img.convert('RGB')
	img = image.pil2tensor(img,np.float32).div_(224)
	img = image.Image(img)

    #display image
	display_img = mpimg.imread(file_path)
	st.image(pil_img,use_column_width=True)

	predict(img)




#Image URL
if options == 'Enter Image URL':
	url = st.text_input('Enter Image URL')

	if url != '':
		try :
			response = requests.get(url)
			pil_img = Image.open(BytesIO(response.content))
			st.image(pil_img,use_column_width=True)
			st.success('Success')

			#read the image
			img = pil_img.convert('RGB')
			img = image.pil2tensor(img,np.float32).div_(224)
			img = image.Image(img)
			predict(img)
			
		except:
			st.error('Invalid URL')



#From  file
if options == 'Upload an image':
	upload_image = st.file_uploader("Choose an image", type = 'jpg')
	if upload_image is not None:
		pil_img=Image.open(upload_image)
		st.image(pil_img,caption = 'UPLOADED IMAGE', use_column_width=True)

	#read the image
		img = pil_img.convert('RGB')
		img = image.pil2tensor(img,np.float32).div_(224)
		img = image.Image(img)
		predict(img)



