from imutils.video import VideoStream
import imutils 
import time
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import argparse
from utils_social_distancing import GetFirstFrame,ScaleImage


def draw_circle(event,x,y,flags,param):
	global pointIndex
	global pts
	global image
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
		pts[pointIndex] = (x,y)
		pointIndex = pointIndex + 1

def show_window(img,name= 'None'): 
	while True:
		cv2.imshow(name, img)
		
		if(pointIndex == 4):
			cv2.waitKey(3000)
			break
		
		if (cv2.waitKey(20) & 0xFF == 27) :
			break

def main():
	global pointIndex
	global pts
	global image
	equal_size = False
	
	# 1. Reading the input video
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, type=str, help="Path Video Input or Camera")
	ap.add_argument("-m", "--machine", required = True, type=str, help="Machine device")
	args = vars(ap.parse_args())
	video_file = args['input']
	machine_device = args['machine']
	print("[INFO] Initializing Video or Camera: ", video_file)

	# 2. Get the first Frame of the video
	filename = './Images/First_Frame_Calibration.jpeg'
	GetFirstFrame(video_file,filename = filename, machine_device = machine_device)

	# 3. Start Calibration
	print("[INFO] Starting Calibration: ")

	# 3.1 Points for Bird Eye View
	print('[INFO] Select points for create the Bird Eye View (clock wise)')
	input('[INFO] Press enter to continue')
	pts = [(0,0),(0,0),(0,0),(0,0)]
	pointIndex = 0
	image = cv2.imread(filename)
	image = ScaleImage(image, equal_size = equal_size, max_size_image = 1000 )
	cv2.namedWindow('Calibration')
	cv2.setMouseCallback('Calibration',draw_circle)
	show_window(image,'Calibration')
	cv2.destroyAllWindows()


	real_width = float(input('Real width distance (ct): '))
	real_height = float(input('Real height distance (ct): '))
	min_distance = float(input('Min distance distance (ct): '))
	if real_width >= real_height:
		pix_width = 250
		print("Pixels output for width: ", pix_width)
		pix_height = int(real_height* pix_width/real_width) 
		scale_tr_distance =  pix_width/real_width
		min_distance_pix = min_distance * scale_tr_distance
		
	else :    

		pix_height = 250
		print("Pixels output for width: ", pix_height)
		pix_width = int(real_width * pix_height/real_height) 
		scale_tr_distance =  pix_height/real_height
		min_distance_pix = min_distance*scale_tr_distance

	print('Min distance ' +str(min_distance) + ' (ct) in pixels: ' + str(min_distance_pix))
	# Saving Files
	bird_eye_view_points = np.asarray(pts ,dtype=np.float32)
	min_distance_pix = np.array(min_distance_pix, dtype = np.float32)
	np.save('./Numpy Files/bird_eye_view_points.npy', bird_eye_view_points)
	np.save('./Numpy Files/min_distance_pix.npy', min_distance_pix)

	# Draw the lines connecting the points to form the trapezoid:
	for cnt,point in enumerate(bird_eye_view_points):
		cv2.circle(image, tuple(point), 4, (148, 0, 211), -1)
		cv2.putText(image, str(cnt + 1), ( min(int(point[0]) - 10,image.shape[1]),min(int(point[1]) +10,image.shape[0])),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	points = bird_eye_view_points.reshape((-1,1,2)).astype(np.int32)
	cv2.polylines(image, [points], True, (148,0,211), thickness=2)
	cv2.imwrite('./Images/Points_Bird_Eye_View.jpeg', image)
	
	# Point for Region of interest (ROI)
	pts = [(0,0),(0,0),(0,0),(0,0)]
	pointIndex = 0
	image = cv2.imread(filename)
	image = ScaleImage(image, equal_size = equal_size, max_size_image = 1000 )
	cv2.namedWindow('ROI')
	cv2.setMouseCallback('ROI',draw_circle)
	show_window(image,'ROI')
	cv2.destroyAllWindows()

	# Saving Files
	roi_points = np.asarray(pts,dtype=np.float32)
	for cnt,point in enumerate(roi_points):
		cv2.circle(image, tuple(point), 4, (255,127,80), -1)
		cv2.putText(image, str(cnt + 1), ( min(int(point[0]) - 10,image.shape[1]),min(int(point[1]) +10,image.shape[0])),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	points = roi_points.reshape((-1,1,2)).astype(np.int32)
	cv2.polylines(image, [points], True, (255,127,80), thickness=2)
	cv2.imwrite('./Images/Points Region_of_Interest.jpeg', image)

	# Saving Image with 2 areas combined
	for cnt,point in enumerate(bird_eye_view_points):
		cv2.circle(image, tuple(point), 4, (148, 0, 211), -1)
		cv2.putText(image, str(cnt + 1), ( min(int(point[0]) - 10,image.shape[1]),min(int(point[1]) +10,image.shape[0])),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	points = bird_eye_view_points.reshape((-1,1,2)).astype(np.int32)
	cv2.polylines(image, [points], True, (148,0,211), thickness=2)
	cv2.imwrite('./Images/Image_with_2_regions.jpeg', image)

	# Calculate the Transformation  Matrix (M)
	outputQuad         = np.float32([[0,0],[pix_width,0],[pix_width,pix_height],[0,pix_height]])
	inputQuad          = bird_eye_view_points
	Image_W,Image_H    = image.shape[1],image.shape[0] 
	inputCorners       = np.float32([[0,0],[Image_W,0],[0,Image_H],[Image_W,Image_H]])
	# Calculate  the First Image Transformation
	M1 = cv2.getPerspectiveTransform(inputQuad,outputQuad)
	outputCorners = cv2.perspectiveTransform(np.array([inputCorners]),M1)
	# Found the Boundaries fof the rectangle
	br = cv2.boundingRect(outputCorners) 
	size_transformation = (br[2],br[3])
	for i in range(0,4):
		outputQuad[i] = outputQuad[i] -1*np.float32([br[0],br[1]])

	# Calculate the final matrix for transformation 
	M2 = cv2.getPerspectiveTransform(inputQuad,outputQuad)
	output = cv2.warpPerspective(image, M2, size_transformation)
	# Save the Image Transformation
	M2 = np.asarray(M2,dtype=np.float32)
	np.save('./Numpy Files/Homolographic_Matrix.npy', M2)
	# Output has the 2 regions	
	cv2.imwrite('./Images/Image_with_transformation_applied.jpeg',output)
	# Image Transformation with ROI
	point_i=[]
	for pt in roi_points:
		point_i.append(pt)
	roi_points = np.array([point_i],dtype ='float32')
	roi_points_transform = cv2.perspectiveTransform(roi_points,M2)
	br_roi = cv2.boundingRect(roi_points_transform)
	np.save('./Numpy Files/roi_points.npy', roi_points)
	output_roi = output[br_roi[1]:(br_roi[1]+br_roi[3]),br_roi[0]:(br_roi[0]+br_roi[2]),:] 
	cv2.imwrite('./Images/Image_with_transformation_applied_ROI.jpeg',output_roi)

	# Show Image with the 2 regions seleted
	img_ = cv2.imread('./Images/Image_with_2_regions.jpeg') 
	plt.figure(figsize=(12, 12))
	plt.imshow( cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
	plt.xticks(np.arange(0, img_.shape[1], step=150),rotation=90)
	plt.yticks(np.arange(0, img_.shape[0], step=150))
	plt.title('Image with regions: Calibration & ROI ')
	plt.show()
	del img_

	# Show Image with the 2 regions seleted and transform
	img_1 = cv2.imread('./Images/Image_with_2_regions.jpeg') 
	img_2 = cv2.imread('./Images/Image_with_transformation_applied.jpeg') 
	img_3 = cv2.imread('./Images/Image_with_transformation_applied_ROI.jpeg') 
	fig = plt.figure(figsize=(8,8))
	a  = fig.add_subplot(1, 2, 1)
	im = plt.imshow( cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
	a.set_title('Raw images with regions')
	plt.xticks(np.arange(0, img_1.shape[1], step=150),rotation=90)
	plt.yticks(np.arange(0, img_1.shape[0], step=150))
	a = fig.add_subplot(1, 2, 2)
	im = plt.imshow( cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
	a.set_title('Bird eye view of the whole image')
	plt.xticks(np.arange(0, img_2.shape[1], step=150),rotation=90)
	plt.yticks(np.arange(0, img_2.shape[0], step=150))
	plt.grid(True, color='g', linestyle='-', linewidth=0.9)
	plt.show()

	# Show Image with the 2 regions seleted and transform with ROI
	
	fig = plt.figure(figsize=(8,8))
	a  = fig.add_subplot(1, 2, 1)
	im = plt.imshow( cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
	a.set_title('Raw images with regions')
	plt.xticks(np.arange(0, img_1.shape[1], step=150),rotation=90)
	plt.yticks(np.arange(0, img_1.shape[0], step=150))
	a = fig.add_subplot(1, 2, 2)
	im = plt.imshow( cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB))
	a.set_title('Bird eye view of the whole image')
	plt.xticks(np.arange(0, img_3.shape[1], step=150),rotation=90)
	plt.yticks(np.arange(0, img_3.shape[0], step=150))
	plt.grid(True, color='g', linestyle='-', linewidth=0.9)
	plt.show()
	del img_1, img_2, img_3

if __name__ == '__main__':
	main()