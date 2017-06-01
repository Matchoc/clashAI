import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import cv2
import numpy
import json
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
	archer_path = os.path.join("data", "ref", "cardarcher.png")
	ss_path = os.path.join("data", "screenshots", "ss-20170524-220603.png")

	archer_img = Image.open(archer_path)
	archer_width, archer_height = archer_img.size
	archer_pixels = numpy.array(archer_img.getdata(), dtype=numpy.uint8)[:,0:3].reshape(archer_height,archer_width,3)
	#print(archer_pixels)
	#print(archer_pixels.shape)
	#print(archer_pixels.dtype)
	hasAlpha = archer_img.mode == "RGBA"
	#btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	
	ss_img = Image.open(ss_path)
	ss_width, ss_height = ss_img.size
	ss_pixels = numpy.array(ss_img.getdata(), dtype=numpy.uint8)[:,0:3].reshape(ss_height,ss_width,3)
	#print(ss_pixels)
	hasAlpha = ss_img.mode == "RGBA"
	
	#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	res = cv2.matchTemplate(ss_pixels,archer_pixels,cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	plt.imshow(res,cmap = 'gray')
	plt.show()
	
	print("min_val : " + str(min_val) + ", max_val : " + str(max_val) + ", min_loc : " + str(min_loc) + ", max_loc : " + str(max_loc))
	