import tensorflow as tf
import cv2
import time
import argparse
import os
import math
import numpy
import posenet
import numpy as np

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure




def bounding_box(coords):

	min_x = 100000 
	min_y = 100000
	max_x = -100000 
	max_y = -100000

	for item in coords:
		if item[0] < min_x:
			min_x = item[0]

		if item[0] > max_x:
			max_x = item[0]

		if item[1] < min_y:
			min_y = item[1]

		if item[1] > max_y:
			max_y = item[1]

	return [(int(min_x),int(min_y)),(int(max_x),int(min_y)),(int(max_x),int(max_y)),(int(min_x),int(max_y))]

def getpoints(image_input,flag,model_black_image):

	with tf.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(101, sess)
		output_stride = model_cfg['output_stride']
		pos_temp_data=[]
		sum = 0
		
		input_image, draw_image, output_scale = posenet.read_imgfile(
			image_input, scale_factor=1.0, output_stride=output_stride)

		heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
			model_outputs,
			feed_dict={'image:0': input_image}
		)

		pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
			heatmaps_result.squeeze(axis=0),
			offsets_result.squeeze(axis=0),
			displacement_fwd_result.squeeze(axis=0),
			displacement_bwd_result.squeeze(axis=0),
			output_stride=output_stride,
			max_pose_detections=1,
			min_pose_score=0.1)

		keypoint_coords *= output_scale

		
		draw_image = posenet.draw_skel_and_kp(flag,
			draw_image, pose_scores, keypoint_scores, keypoint_coords,
			min_pose_score=0.1, min_part_score=0.0001)
		
		black_image = numpy.zeros((draw_image.shape[0],draw_image.shape[1],3),dtype='uint8')
		
		if flag ==1:

			black_image = posenet.draw_skel_and_kp(flag,
				black_image, pose_scores, keypoint_scores, keypoint_coords,
				min_pose_score=0.1, min_part_score=0.0001)
		if flag ==0:
			black_image = posenet.draw_skel_and_kp(flag,
				model_black_image, pose_scores, keypoint_scores, keypoint_coords,
				min_pose_score=0.1, min_part_score=0.0001)

			
		for pi in range(len(pose_scores)):
			if pose_scores[pi] == 0.:
				break
			for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
				
				
				pos_temp_data.append(c[1])
				pos_temp_data.append(c[0])
			for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
				pos_temp_data.append(s)
				sum = sum + s
			pos_temp_data.append(sum)
		
	return pos_temp_data, draw_image, black_image 

def get_roi(points,image):	
	fun_bound = bounding_box(points)
	roi = image[fun_bound[0][1]:fun_bound[2][1], fun_bound[0][0]:fun_bound[1][0]]
	return roi,fun_bound 



def get_new_coords(coords,fun_bound):
	coords[:,:1] = coords[:,:1] - fun_bound[0][0]
	coords[:,1:2] = coords[:,1:2] - fun_bound[0][1]
	return coords

def roi(imagepoints,drawn_image):
	#imagepoints, drawn_image = getpoints(image)
	coords_new_reshaped = imagepoints[0:34]
	coords_new = np.asarray(coords_new_reshaped).reshape(17,2)
	roi,roi_coords = get_roi(coords_new,drawn_image)
	coords_new = get_new_coords(coords_new, roi_coords)
	coords_new = coords_new.reshape(34,)
	coords_new = np.concatenate((coords_new[0:34],imagepoints[34:52]))
	#print(coords_new)
	return roi,coords_new


def cos(model_points, input_points):
	model_points = model_points[0:34]
	input_points = input_points[0:34]
	norm_model = np.linalg.norm(model_points)
	norm_input = np.linalg.norm(input_points)
	model_points = model_points/norm_model
	input_points = input_points/norm_input
	
	norm_model = np.linalg.norm(model_points)
	norm_input = np.linalg.norm(input_points)
	
	model_points = numpy.asarray(model_points).reshape(34,)
	input_points = numpy.asarray(input_points).reshape(34,)
		
	dot = np.dot(model_points,input_points)
	cosine = dot/norm_model*norm_input
	result = (2 * (1 - cosine)) ** (1/2)
	return cosine,result

def weighted(model_points, input_points):
	vector1PoseXY = model_points[0:34]
	vector1Confidences = model_points[34:51]
	vector1ConfidenceSum = model_points[51]

	vector2PoseXY = input_points[0:34]

	summation1 = 1/vector1ConfidenceSum

	summation2 = 0
	for i in range(len(vector1PoseXY)):
		tempConf = math.floor(i/2)
		tempSum = vector1Confidences[tempConf] * abs(vector1PoseXY[i] - vector2PoseXY[i])
		summation2 = summation2 + tempSum

	return (summation1 * summation2)

def create_gif(image_stack,title_stack, file_name, title_str='title'):
	'''
	image_stack :: 3-D array containing images
	title_stack :: title per frame
	file_name   :: output filename
	title_str   :: title description string

	Creates matplotlib dependent animation
	'''
	fig, ax=plt.subplots(figsize=(12,5))
	container = []
	#
	for image, title in zip(image_stack, title_stack):
		ims = ax.imshow(image)
		ts = ax.text(0.5,1.05,"Score : {}%".format(title), ha="center", transform=ax.transAxes)
		ax.axis('off')
		container.append([ims,ts])
    #
	ani = animation.ArtistAnimation(fig, container, interval=50, blit=True)
	ani.save(file_name)

def percentage_score(score):
	# new_score = 1 - score
	percentage =  100 - (score* 100)
	return int(percentage)



plot_name = 'save'
filename = 'testplots'


fig = plt.figure()




model_image = "model_punch.png"
model_image_frame = cv2.imread(model_image)
model_points,model_drawn_image,model_black_image = getpoints(model_image_frame,1,None)
model_drawn_image = cv2.cvtColor(model_drawn_image, cv2.COLOR_BGR2RGB)
model_roi, model_new_coords = roi(model_points,model_drawn_image)
cap = cv2.VideoCapture("alwin_punch_2.mp4")
totalframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for_plot=np.zeros((int(totalframe),4180,7888,3),dtype='uint8')
model_black_image = cv2.resize(model_black_image,(230,330))
black_image_horizontal = numpy.zeros((330,72,3),dtype='uint8')
black_image_vertical = numpy.zeros((95,302,3), dtype='uint8') 
temp2 = np.concatenate((black_image_horizontal,model_black_image),axis = 1)
temp3 = np.concatenate((black_image_vertical,temp2), axis = 0)

i = 0
ts = []
time1 = []
if cap.isOpened() is False:
	print("error in opening video")
while cap.isOpened():
	ret_val, image = cap.read()
	if i<90:
	#if ret_val:
		start = time.time()
		
	
		image = cv2.resize(image,(372,495))
		
		model_black_image_crop = temp3[0:int(model_black_image.shape[0]), 17:int(model_black_image.shape[1])]
		
		#model_points,model_drawn_image = getpoints(model_image)
		input_points,input_drawn_image, input_black_image = getpoints(image,0,model_black_image_crop)
		input_roi, input_new_coords = roi(input_points,input_drawn_image)

		#print(model_new_coords)

		#result = weighted(model_new_coords, input_new_coords)
		cosine,dresult = cos(model_new_coords, input_new_coords)

		instantaneous_score = percentage_score(dresult)
		#print(result)
		ts.append(instantaneous_score)
		# time1.append(time.time())
		time1.append(i)
		if dresult < 0.2:
			cv2.putText(input_black_image, 'Pose Match!!',(15,35),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			# cv2.putText((input_black_image,'Pose Match!!',(input_black_image.shape[0]+ 10,input_black_image[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,255,0),2,cv2.LINE_AA))

		elif dresult > 0.2:
			#cv2.puttext(font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(input_black_image, 'Pose Not Matching!!',(15,35),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		plt.ylim(60, 100)
		plt.xlim(0,105)
		plt.xlabel("Frames")
		plt.ylabel("Score")
		plt.plot(time1,ts)
		#plt.canvas.draw()
		plt.tight_layout()


		#plt.show()
		plt.savefig('test_plot.png')
		plot= cv2.imread("test_plot.png")


		temp = []

		#crop_img = input_black_image[80:int(input_black_image.shape[0])-70, 0:int(input_black_image.shape[1])-30]
		model_black_image = cv2.resize(model_black_image, (1972,2980))
		plot = cv2.resize(plot, (7888,1200))
		model_drawn_image = cv2.resize(model_drawn_image, (1972,2980))
		input_black_image = cv2.resize(input_black_image, (1972,2980))
		input_drawn_image = cv2.resize(input_drawn_image, (1972,2980))
		# model_drawn_image = cv2.cvtColor(model_drawn_image, cv2.COLOR_BGR2RGB)
		input_drawn_image = cv2.cvtColor(input_drawn_image, cv2.COLOR_BGR2RGB)
		print(input_drawn_image.shape)
		temp = np.concatenate((model_drawn_image,input_drawn_image,model_black_image,input_black_image),axis = 1)
		for_plot[i] = np.concatenate((temp,plot),axis = 0)
		i = i + 1
		print(i)
	else:
		break
cap.release()

create_gif(for_plot,ts, 'test_kr_imposed3.mp4', title_str='Cumulative Score :: ')









