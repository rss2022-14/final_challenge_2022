
import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	largest_area = 0
	bounding_box = ((0,0),(0,0))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert BGR to HSV format
	lower_range = np.array([5,120,120])
	upper_range = np.array([18,255,255])
	mask = cv2.inRange(img.copy(),lower_range,upper_range)
	#ret, thresh = cv2.threshold(img, 16, 255, 0)

	contours, hierarcy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	#contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for i in range(len(contours)):
		x,y,w,h = cv2.boundingRect(contours[i])
		area = w*h
		if area > largest_area:
			largest_area = area
			bounding_box = ((x,y),(x+w,y+h))
	#print(img)
	#print(img.shape)
	#image_print([img])

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box

#img = cv2.imread(cv2.samples.findFile("test1.jpg"))
#image_print(img)
