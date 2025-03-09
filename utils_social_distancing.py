import os
import cv2
import numpy as np
import datetime
import argparse

def Object_Detection_Arq(model_name = 'YoloV3',path = './Models'):
	'''
	Input
	model_name : Model name for Object Detection
	path       : Path of the weights, cfg, and coco.names files

	Output:
	Yolov3_net  : Yolov3 Network
	OutputLayers: Output layers of the yolov3 model
	Classes     : Coco Classes
	'''

	if model_name == 'YoloV3':
		classes    = []
		weights    = os.path.join(path,'yolov3.weights')
		cfg        = os.path.join(path,'yolov3.cfg')
		coco_names = os.path.join(path,'coco.names')
		print(weights)
		print(cfg)
		print(coco_names)
		Yolov3_net = cv2.dnn.readNet(weights,cfg) 
		with open(coco_names,"r") as f:
			classes = [line.strip() for line in f.readlines()]
		
		layer_names = Yolov3_net.getLayerNames()
		OutputLayers = [layer_names[i[0] - 1] for i in Yolov3_net.getUnconnectedOutLayers()]  

	return Yolov3_net,OutputLayers,classes

def  gstreamer_pipeline(
	capture_width=1280,
	capture_height=720,
	display_width=1280,
	display_height=720,
	framerate=60,
	flip_method=6,):
	return (
		"nvarguscamerasrc ! "
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, "
		"format=(string)NV12, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink"
		% (
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
)

def GetFirstFrame(video_file,filename='./Images/First_Frame_Calibration.jpeg', machine_device = 'jetson'):
	'''
	Input:
	video_file : Video file name
	filename   : File with the saved calibration image
	'''
	if video_file == '0' and machine_device == 'jetson':
		vidcap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
	else:
		vidcap = cv2.VideoCapture(video_file)
	success, image = vidcap.read()
	if success:
		cv2.imwrite(filename, image)


def ScaleImage(image, equal_size = False, max_size_image = 1000 ):
	'''
	Input:
	image          : Input Image
	equal_size     : True -> Same Image / False -> Scale Image
	max_size_image : Max size of (Height, Width)
	'''


	Image_H, Image_W = image.shape[0],image.shape[1]
	# Scale the image
	if  equal_size:
		return image
	else :
		scale = max_size_image / max (Image_H, Image_W)
		new_W , new_H = int(Image_W*scale), int(Image_H*scale)
		img_resize = cv2.resize(image,(new_W,new_H))
		return img_resize



def blob_from_image(image, target_size):
	"""
	This function creates a blob from the image or video frame
	to be predicted, scales [0-255] and resizes it
	as required by the YOLOv3 model
	Input:
		image: RGB Image
		target_size: Target dimension (416 x 416)
	Output:
		Returns a blob from image      
	"""
	if not isinstance(target_size, tuple): 
		raise Exception("target_size must be a tuple (width, height)")

	blob = cv2.dnn.blobFromImage(image, 
								 1/255.,
								 target_size,
								 [0,0,0],
								 1,
								 crop=False)
	
	return blob


def predict(blob, model, output_layers):
	"""
	This function performs the prediction to detect people in the scene,
	returns the class and bounding box values
	Input:
		blob: Blob from the original image
		model: object detection model (e.g. Yolo3)
		output_layers: Frame layers to predict based on the blob image
	
	Output:
		Return classes and boxes from image
	"""
	model.setInput(blob)
	outputs = model.forward(output_layers)

	return outputs

def non_maximum_suppression(image, outputs, confidence_threshold=0.6, nms_threshold=0.4):
	"""
	This function performs non-maximum suppression, eliminating overlapping bounding
	boxes based on the suppression threshold and applies a confidence
	threshold to accept object detections
	Input:  
		image: Original image or video frame  
		outputs: Prediction made by the 'predict' method  
		confidence_threshold: Confidence threshold to accept a detection  
		nms_threshold: Threshold for non-maximum suppression elimination  

	Output:  
		Returns three values:  
		1. The new bounding boxes where people are located.  
		2. The indices of the bounding boxes that meet the confidence threshold.  
		3. The indices of the classes to which each object belongs.  
	"""
	class_ids = []
	confidences = []
	boxes = []

	img_height, img_width = image.shape[:2]
	
	# Detecting bounding boxing
	for output in outputs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > confidence_threshold:
				cx = int(detection[0] * img_width)
				cy = int(detection[1] * img_height)
				width = int(detection[2] * img_width)
				height = int(detection[3] * img_height)
				left = int(cx - width / 2)
				top = int(cy - height / 2)
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
	
	nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
	
	return boxes, nms_indices, class_ids,confidences


def get_domain_boxes(classes, class_ids, nms_indices, boxes, domain_class):
	"""
	This function obtains the final bounding boxes that match our specific domain class, in this case, it must be 'person'
	Input:  
		classes: Original COCO Dataset classes used to train the YOLOv3 model  
		class_ids: Class indices obtained after applying non-maximum suppression  
		nms_indices: Indices of the bounding boxes after applying non-maximum suppression  
		boxes: Final bounding boxes after applying non-maximum suppression  
		domain_class: Target class for which we want the final bounding boxes  

	Output:  
		Returns the final list of bounding boxes for the desired class. Each bounding box  
		contains the coordinates (left, top), width and height (width, height), and the center point  
		of the bounding box (cx, cy)  
	"""

	domain_boxes = []
	for index in nms_indices:
		idx = index[0]
		class_name = classes[class_ids[idx]]
		if class_name in domain_class:
			box = boxes[idx]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			cx = left + int(width / 2)
			cy = top + int(height / 2)
			domain_boxes.append((left, top, width, height, cx, cy))
	
	return domain_boxes



def __matrix_bird_eye_view(A):
	"""
	This function returns the homography matrix obtained in previous steps.
	"""
	return A

def eucledian_distance(point1, point2):
	"""
	This function calculates the Euclidean distance between a pair of points
	Input:
		point1: First point
		point2: Second Point
	Output:
		Ecluedian distance between both points
	"""
	x1,y1 = point1
	x2,y2 = point2
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def draw_new_image_with_boxes(image, people_good_distances, people_bad_distances, distance_allowed, draw_lines=False):
	"""
	This function draws bounding boxes and lines between instances of the same type to better
	understand their distance.
	Input:  
		image: Original image on which the bounding boxes and lines will be drawn  
		people_good_distances: List of bounding boxes that respect social distancing  
		people_bad_distances: List of bounding boxes that do not respect social distancing  
		distance_allowed: Minimum allowed value for social distancing  
		draw_lines: Flag (True/False) to draw a line between two points, only for people who do not respect the allowed social distance.  
	Output:  
    	Returns the new image with bounding boxes drawn and lines if they were enabled.  
	"""
	green = (0, 255, 0)
	red = (255, 0, 0)
	new_image = image.copy()

	for person in people_bad_distances:
		left, top, width, height = person[:4]
		cv2.rectangle(new_image, (int(left), int(top)), (int(left + width), int(top + height)), red, 2)
	
	for person in people_good_distances:
		left, top, width, height = person[:4]
		cv2.rectangle(new_image, (int(left), int(top)), (int(left + width), int(top + height)), green, 2)
	
	if draw_lines:
		for i in range(0, len(people_bad_distances)-1):
			for j in range(i+1, len(people_bad_distances)):
				cxi,cyi,bevxi,bevyi = people_bad_distances[i][4:]
				cxj,cyj,bevxj,bevyj = people_bad_distances[j][4:]
				distance = eucledian_distance([bevxi, bevyi], [bevxj, bevyj])
				if distance < distance_allowed:
					cv2.line(new_image, (int(cxi), int(cyi)), (int(cxj), int(cyj)), red, 2)
			
	return new_image

def __map_points_to_bird_eye_view(points,A):
	"""
	This function maps points from the original view to the Bird Eye view
	Input:  
    	points: Two-dimensional list of points to be transformed to Bird Eye view  

	Output:  
    	Returns the new points belonging to the Bird Eye view  
	"""
	if not isinstance(points, list):
		raise Exception("points must be a list of type [[x1,y1],[x2,y2],...]")
	
	matrix_transformation = __matrix_bird_eye_view(A)
	new_points = np.array([points], dtype=np.float32)
	
	return cv2.perspectiveTransform(new_points, matrix_transformation)


def generate_bird_eye_view(good, bad ,warp_size,desire_size,br=[]):
	'''
	desire_size : (Height, Width)
	'''
	red = (255,0,0)
	green = (0,255,0)

	# Background size
	bird_eye_view = np.zeros((warp_size[0], warp_size[1], 3), dtype=np.uint8)
	# Points that respect the distance
	for point in good:
		point = point[6:]
		point = [int(point[0]), int(point[1])]
		cv2.circle(bird_eye_view, tuple(point), 8, green, -1)
	
	flagBad = False

	# Points that don't respect the distance
	for point in bad:
		flagBad = True
		point = point[6:]
		point = [int(point[0]), int(point[1])]
		cv2.circle(bird_eye_view, tuple(point), 8, red, -1)
	

	if flagBad:
		now=datetime.datetime.now()
		print("[INFO] Social Distance incident found at : " + now.isoformat())

	# ROI of bird eye view
	if len(br)>0:
		bird_eye_view = bird_eye_view[br[1]:(br[1]+br[3]),br[0]:(br[0]+br[2]),:]	
	# Bird Eye View resize
	bird_eye_view_resize = cv2.resize(bird_eye_view, (desire_size[1],desire_size[0]))

	return bird_eye_view_resize

def generate_bird_eye_view_2(good, bad ,warp_size,desire_size,br=[]):
	'''
	desire_size : (Height, Width)
	'''
	print("OK generate_bird_eye_view_2")
	red = (255,0,0)
	green = (0,255,0)
	scale_w = np.float32(desire_size[1]) / warp_size[1]
	scale_h = np.float32(desire_size[0]) / warp_size[0]
	bird_eye_view = np.zeros((desire_size[0], desire_size[1], 3), dtype=np.uint8)
	# Points that respect the distance
	for point in good:
		point = point[6:]
		point = [int(point[0] * scale_w), int(point[1] * scale_h)]
		cv2.circle(bird_eye_view, tuple(point), 10, green, -1)
	
	flagBad = False
	# Points that don't respect the distance
	for point in bad:
		flagBad = True
		point = point[6:]
		point = [int(point[0] * scale_w), int(point[1] * scale_h)]
		cv2.circle(bird_eye_view, tuple(point), 10, red, -1)

	if flagBad:
		now=datetime.datetime.now()
		print("[INFO] Social Distance Incident found at: " + now.isoformat())

	# ROI of bird eye view
	if len(br) > 0:
		cut_y1 = br[1]
		cut_y2 = br[1] + br[3] 
		cut_x1 = br[0] 
		cut_x2 = br[0] + br[2]

		ncut_x2 = int(cut_x2 * scale_w)
		ncut_x1 = int(cut_x1 * scale_w)
		ncut_y2 = int(cut_y2 * scale_h)
		ncut_y1 = int(cut_y1 * scale_h)

		bird_eye_view = bird_eye_view[ncut_y1:ncut_y2,ncut_x1:ncut_x2]
	return bird_eye_view



def people_distances_bird_eye_view(boxes, distance_allowed,A):
	"""
	This function detects if people are respecting social distancing  
	Input:  
		boxes: Bounding boxes obtained after applying the get_domain_boxes function  
		distance_allowed: Minimum allowed distance to determine if a person respects social distancing or not  
	Output:  
		Returns a tuple containing 2 lists:  
		1. The first list contains information about the points (people) who respect social distancing.  
		2. The second list contains information about the points (people) who do not respect social distancing. 
	"""
	people_bad_distances = []
	people_good_distances = []
	# Taking center and bottom values
	result = __map_points_to_bird_eye_view([[box[4],box[1]+box[3]] for box in boxes],A)[0]
	# We create new bounding boxes with values mapped from the bird's eye view (8 elements per item)
	# left, top, width, height, cx, cy, bev_cy, bev_cy
	new_boxes = [box + tuple(result) for box, result in zip(boxes, result)]

	for i in range(0, len(new_boxes)-1):
		for j in range(i+1, len(new_boxes)):
			cxi,cyi = new_boxes[i][6:]
			cxj,cyj = new_boxes[j][6:]
			distance = eucledian_distance([cxi,cyi], [cxj,cyj])
			if distance < distance_allowed:
				people_bad_distances.append(new_boxes[i])
				people_bad_distances.append(new_boxes[j])

	people_good_distances = list(set(new_boxes) - set(people_bad_distances))
	people_bad_distances = list(set(people_bad_distances))
	
	return (people_good_distances, people_bad_distances)

def str2bool(v):
	'''
	This function converts string inputs to boolean values.
	'''
	if isinstance(v, bool):
		return v
	elif v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')