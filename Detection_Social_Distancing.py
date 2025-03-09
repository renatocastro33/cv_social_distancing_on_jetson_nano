import time
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import threading
import argparse
import datetime
from utils_social_distancing import *

def birdeyeviewShow(min_distance, net, outputlayers, classes ,HM, roi_points, br_roi, warp_shape, recordFlag, videoWriter, videoWriter2 ):
	global frameToBirdEyeView
	global	frame_id
	global ret
	frame = 0 
	while(True):
		if frame_id != 0:
			try:
				notEqualFrames = (frame == frameToBirdEyeView).all()
			except Exception as e:
				notEqualFrames = (frame == frameToBirdEyeView)
			
			if not (notEqualFrames):
				a = time.time()
				frame = frameToBirdEyeView
				blob = blob_from_image(frame, (320, 320))
				# We obtain the detection bounding boxes:
				outputs = predict(blob,net,outputlayers)
				# We apply Non-Maximum Suppression to improve detection:
				boxes, nms_boxes, class_ids,confi = non_maximum_suppression(frame, outputs, confidence_threshold=0.6,nms_threshold=0.4)
				# We obtain the detection bounding boxes only for the 'person' class:
				person_boxes = get_domain_boxes(classes, class_ids, nms_boxes, boxes, domain_class='person')
				if len(person_boxes) > 0:
					print("[INFO] N° de personas Detectadas en el video: ", len(person_boxes))
					good, bad = people_distances_bird_eye_view(person_boxes, min_distance,HM)
					# We generate the final image with the distance evaluation:
					frame = draw_new_image_with_boxes(frame, good, bad, min_distance, draw_lines=True)
					# We generate the 'Bird's Eye View' with the points evaluated by distance:
					bird_eye_view = generate_bird_eye_view(good, bad ,warp_shape,(400,800),br = br_roi)
					print(videoWriter)
					print(videoWriter2)
					frame = cv2.resize(frame, (640,380))
					bird_eye_view = cv2.resize(bird_eye_view, (640,380))
					cv2.imshow('Social Distancing In Past Visualization', frame)
					cv2.imshow('Social Distancing Birdeyeview: ', bird_eye_view)
					if recordFlag:
						videoWriter.write(frame)
						videoWriter2.write(bird_eye_view)
				b = time.time()
				print(b-a)
				if ret == False:
					break
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


def VideoShow(video, numberDropFrames, noDropFlag):
	
	global frameToBirdEyeView
	global frame_id
	global ret
	equal_size = False
	frame_id = 0
	while True:
		ret, frame = video.read()
		if ret == False:
			break	
		frame = ScaleImage( frame, equal_size)
		if frame_id % numberDropFrames == 0 or noDropFlag:
			print("[INFO] Birdeyeview Starting ...")
			frameToBirdEyeView = frame
		frame_id += 1

		cv2.imshow('Social Distancing: ', frame) 

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



def main():
	# 1. Reading the input video
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, type=str, help="Path Video Input or Camera")
	ap.add_argument("-m", "--machine", required = True, type=str, help="Machine device")
	ap.add_argument("-r", "--record",  required = True, type=str2bool, help="Flag record video")
	ap.add_argument("-d", "--noDrop", nargs='?', required = False, default = True, type=str2bool, help="Flag no drop frames")
	ap.add_argument("-n", "--numberDropFrames",nargs='?', required = False, default = 1, type=str, help="Number of iterations to get an incident")
	
	args = vars(ap.parse_args())
	video_file = args['input']
	machine_device = args['machine']
	recordFlag = args['record']
	noDropFlag = args['noDrop']
	numberDropFrames = int(args['numberDropFrames'])

	print("[INFO] Starting Videofile or Camera: ", video_file)
	if recordFlag:
		print("[INFO] Loading VideoWriter for recording the social distancing detection.")	
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		fourcc1 = cv2.VideoWriter_fourcc(*'DIVX')
		now=datetime.datetime.now()
		videoWriter = cv2.VideoWriter('./RecordedVideos/Record_Detection_' + now.isoformat() +'.avi', fourcc, 10, (640,380), True)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,500)
		fontScale              = 1
		fontColor              = (255,255,255)
		lineType               = 2

		print("[INFO] Loading VideoWriter for recording the bird's eye view")
		videoWriter2 = cv2.VideoWriter('./RecordedVideos/Record_Birdeyeview_' + now.isoformat() +'.avi', fourcc1, 10, (640,380), True)


	if video_file == '0' and machine_device == 'jetson':
		video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
	else:
		video = cv2.VideoCapture(video_file)

	# 2. Reading all the needed files
	min_distance = np.load('./Numpy Files/min_distance_pix.npy')
	net, outputlayers, classes = Object_Detection_Arq(model_name = 'YoloV3',path = './Models')
	HM = np.load('./Numpy Files/Homolographic_Matrix.npy')
	roi_points = np.load('./Numpy Files/roi_points.npy')
	new_roi = cv2.perspectiveTransform(roi_points,HM)
	br_roi = cv2.boundingRect(new_roi)
	warp_shape = cv2.imread('./Images/Image_with_transformation_applied.jpeg').shape
	print("[INFO] Warp Shape: ", warp_shape)

	if recordFlag:
		print("[INFO] Incializando el primer programa")
		t1 = threading.Thread(target=VideoShow, args=(video, numberDropFrames, noDropFlag))
		print("[INFO] Incializando el segundo programa")
		t2 = threading.Thread(target=birdeyeviewShow, args=(min_distance, net, outputlayers, classes ,HM, roi_points, br_roi,warp_shape, recordFlag,videoWriter, videoWriter2))
		print("[INFO] Start the first program")
		t1.start()
		print("[INFO] Start the second program")
		t2.start()
	else:
		t1 = threading.Thread(target=VideoShow, args=(video, numberDropFrames, noDropFlag))
		t2 = threading.Thread(target=birdeyeviewShow, args=(min_distance, net, outputlayers, classes ,HM, roi_points,br_roi, warp_shape,  recordFlag, 0, 0))
		t1.start()
		t2.start()

	t1.join()
	t2.join()

	video.release()
	videoWriter.release
	videoWriter2.release
	cv2.destroyAllWindows()		

if __name__ == '__main__':
	main()

	