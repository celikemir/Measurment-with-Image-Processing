# USAGE
#########################
#    EMİRHAN ÇELİK	#
#	07.11.2018   	#	
#	*Ilmenau*	#
#			#
#########################
# python contour_only.py --image images/coins_01.png

# import the necessary packages

from __future__ import print_function
from scipy.spatial import distance as dist
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import cv2
import numpy as np
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


def gri(croped):
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	thresh3 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	thresh3 = cv2.erode(thresh3, None, iterations=4)
	p = int(image.shape[1] * 0.05)
	thresh3[:, 0:p] = 0
	thresh3[:, image.shape[1] - p:] = 0
	cnts = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		# compute the bounding box of the contour and use the contour to
		# compute the aspect ratio and coverage ratio of the bounding box
		# width to the width of the image
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		crWidth = w / float(gray.shape[1])
		print(c)
		# check to see if the aspect ratio and coverage width are within
		# acceptable criteria
		if w>h:
			# pad the bounding box since we applied erosions and now need
			# to re-grow it
			
			extLeft = tuple(c[c[:, :, 0].argmin()][0])
			extRight = tuple(c[c[:, :, 0].argmax()][0])
			extTop = tuple(c[c[:, :, 1].argmin()][0])
			extBot = tuple(c[c[:, :, 1].argmax()][0])
			#cv2.circle(image, extLeft, 6, (0, 0, 255), -1)
			# extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
			roi = image[y:y + h, x:x + w].copy()
			cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
			break
	return extLeft,extRight,extTop,extBot

def kırmızı(croped):
		redLower = (50, 50, 50)
		redUpper = (199, 178, 149)

	# find the colors within the specified boundaries and apply
	# the mask
		blurred = cv2.GaussianBlur(croped, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		#mask = cv2.inRange(image, lower, upper)
		mask = cv2.inRange(hsv, redLower, redUpper)
		mask = cv2.erode(mask, None, iterations=1)
		mask = cv2.dilate(mask, None, iterations=1)
		#output = cv2.bitwise_and(image, image, mask = mask)
		cv2.imshow("",mask)		
		cnts2 = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts2 = cnts[0] if imutils.is_cv2() else cnts[1]
		c = max(cnts2, key=cv2.contourArea)

	# determine the most extreme points along the contour
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
		cv2.circle(image, extLeft, 6, (0, 0, 255), -1)
		cv2.circle(image, extRight, 6, (0, 255, 0), -1)
		cv2.circle(image, extTop, 6, (255, 0, 0), -1)
		cv2.circle(image, extBot, 6, (255, 255, 0), -1)
	# show the images
		#cv2.imshow("images", np.hstack([image, output]))
		
		return cnts2 




def mavi(croped):
		blueLower = (50, 50, 50)
		blueUpper = (199, 178, 149)

	# find the colors within the specified boundaries and apply
	# the mask
		blurred = cv2.GaussianBlur(croped, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		#mask = cv2.inRange(image, lower, upper)
		mask = cv2.inRange(hsv, blueLower, blueUpper)
		mask = cv2.erode(mask, None, iterations=1)
		mask = cv2.dilate(mask, None, iterations=1)
		#output = cv2.bitwise_and(image, image, mask = mask)
		cv2.imshow("",mask)		
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		c = max(cnts, key=cv2.contourArea)

	# determine the most extreme points along the contour
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
		cv2.circle(image, extLeft, 6, (0, 0, 255), -1)
		cv2.circle(image, extRight, 6, (0, 255, 0), -1)
		cv2.circle(image, extTop, 6, (255, 0, 0), -1)
		cv2.circle(image, extBot, 6, (255, 255, 0), -1)
	# show the images
		#cv2.imshow("images", np.hstack([image, output]))
		
		return extLeft,extRight,extTop,extBot 


def crop(image):
	
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	## (2) Find the target yellow color region in HSV
	hsv_lower = (0, 0, 0)
	hsv_upper = (50 , 50, 50)
	mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

	## (3) morph-op to remove horizone lines
	kernel = np.ones((5,1), np.uint8)
	mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)


	## (4) crop the region
	ys, xs = np.nonzero(mask2)
	ymin, ymax = ys.min(), ys.max()
	xmin, xmax = xs.min(), xs.max()

	croped = image[ymin:(ymax-10), xmin:(xmax-10)]
	return croped


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged



image = cv2.imread(args["image"])
#shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
shifted=image
croped=crop(image)

extLeft,extRight,extTop,extBot=gri(croped)

extLeft2,extRight2,extTop2,extBot2=mavi(croped)
print(gri)
cv2.line(image,extBot,extLeft2,(255,0,0),5)
cv2.line(image,extRight,extRight2,(130,255,130),5)
cv2.line(image,extBot,extLeft,(0,255,0),5)
cv2.line(image,extLeft2,extRight2,(0,0,255),5)
dA = dist.euclidean(extBot, extTop2)
cv2.putText(image, "{:.1f}mm".format(dA),
		(int(extBot[0] - 15), int(extLeft2[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
print(dA)
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding

gray = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = auto_canny(gray)
kernel = np.ones((5,5),np.uint8)
#thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
#thresh=cv2.bitwise_not(thresh)
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)### closing operant
#thresh = cv2.dilate(thresh, None, iterations=2)
#thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("Thresh", thresh)
#sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
mask = np.zeros(image.shape[:2], dtype = "uint8")


# find contours in the thresholded image
__,cnts,hi = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]



#l = []
#for h in hi[0]:
#	if h[0] > -1 and h[2] > -1:
#		l.append(h[2])
#		print(l)	

#cv2.imshow('img2', croped) 


#for i,j in enumerate(k):
# if k[i]-k[i-1]<=3 and k[i]-k[i-2]<=6
#	l.append(k[i])
#
#

print("[INFO] {} unique contours found".format(len(cnts)))

# loop over the contours
#idx = 0
for (i, c) in enumerate(cnts):
	# draw the contour
	
#	x,y,w,h = cv2.boundingRect(c)
#	if w>200 and h>200:
#		idx+=1
#		new_img=image[y:y+h,x:x+w]
	((x, y), _) = cv2.minEnclosingCircle(c)
	#cv2.drawContours(croped, [c], -1, (0, 255, 0), 2)
# show the output image
cv2.imshow("orjinal",image)
cv2.imshow("croped", image)
cv2.waitKey(0)
