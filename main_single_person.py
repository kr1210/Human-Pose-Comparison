import cv2
import time
import numpy as np
import sys
import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import matplotlib.pyplot as plt


protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


def getPoints(frame):
	#frame = cv2.imread("test6.jpg")
	frameCopy = np.copy(frame)
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]
	threshold = 0.1

	net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

	t = time.time()
	# input image dimensions for the network
	inWidth = 368
	inHeight = 368
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
	                          (0, 0, 0), swapRB=False, crop=False)

	net.setInput(inpBlob)

	output = net.forward()
	print("time taken by network : {:.3f}".format(time.time() - t))

	H = output.shape[2]
	W = output.shape[3]

	# Empty list to store the detected keypoints
	points = []
	for i in range(nPoints):
	    # confidence map of corresponding body's part.
	    probMap = output[0, i, :, :]

	    # Find global maxima of the probMap.
	    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
	    
	    # Scale the point to fit on the original image
	    x = (frameWidth * point[0]) / W
	    y = (frameHeight * point[1]) / H

	    if prob > threshold : 
	        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
	        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

	        # Add the point to the list if the probability is greater than the threshold
	        points.append((int(x), int(y)))
	    else :
	        points.append((0,0))
	#fh = open("test"+".txt","w")
	'''for l in range(0,len(points)):
		if points[l] == None:
			points[l] = (0,0)
		'''	
	 #   fh.write((l.x) + " " + str(l.y) + "\n")  
	  
	out = np.asarray(points)
	return out



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pose_match")
#json_data_path = 'data/json_data/'
#images_data_path = 'data/image_data/'

#son_data_path = 'data/json_data/'
#images_data_path = 'data/image_data/'

'''
-------------------------------- SINGLE PERSON -------------------------------------------
Read openpose output and parse body-joint points into an 2D arr ay of 18 rows
Elke entry is een coordinatenkoppel(joint-point) in 3D , z-coordinaat wordt nul gekozen want we werken in 2D
'''





#model = "model5"
#input = "foto2"
#model_json = json_data_path + model + '.json'
#input_json = json_data_path + input + '.json'

model_image = 'test6.jpg'
input_image = 'test6.png'
'''
model_frame = cv2.imread(model_image)
modelFrameCopy = np.copy(model_frame)
modelFrameWidth = model_frame.shape[1]
modelFrameHeight = model_frame.shape[0]
threshold = 0.1

input_frame = cv2.imread(input_image)
inputFrameCopy = np.copy(input_frame)
inputFrameWidth = input_frame.shape[1]
inputFrameHeight = input_frame.shape[0]
'''

model_frame = cv2.imread(model_image)
input_frame = cv2.imread(input_image)
model_features = getPoints(model_frame)
input_features = getPoints(input_frame)
print(model_features)
print(input_features)

'''
Calculate match fo real (incl. normalizing)
'''
#TODO: edit return tuple !!
match_result = pose_match.single_person(model_features, input_features, True)
logger.info("--Match or not: %s  score=%f ", str(match_result.match_bool), match_result.error_score)


'''
Calculate match + plot the whole thing
'''
# Reload features bc model_features is a immutable type  -> niet meer nodig want er wordt een copy gemaalt in single_psoe()
# and is changed in single_pose in case of undetected bodyparts
#model_features = parse_openpose_json.parse_JSON_single_person(model_json)
#input_features = parse_openpose_json.parse_JSON_single_person(input_json)
pose_match.plot_single_person(model_features, input_features, model_image, input_image)

plt.show()


