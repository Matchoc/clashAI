import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import glob
import win32gui
import win32ui
import win32con
import win32api
import datetime
import dateutil.relativedelta
import operator
import numpy
import json
import scipy.ndimage
import multiprocessing
import matplotlib.pyplot as plt
import cv2
from PIL import Image

PRINT_LEVEL=4
DATA_FOLDER = "data"
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))
		
class ScopedTimer:
	totals = {}
	def __init__(self, name, level=3):
		self.starttime = datetime.datetime.now()
		self.name = name
		self.level = level
		
	def __del__(self):
		delta = datetime.datetime.now() - self.starttime
		if self.name not in ScopedTimer.totals:
			ScopedTimer.totals[self.name] = datetime.timedelta(0)
		ScopedTimer.totals[self.name] += delta
		myprint("{name} : {delta} / {total}".format(name=self.name, delta=str(delta), total=str(ScopedTimer.totals[self.name])), self.level)
		#myprint(str(self.name) + " : " + str(delta),self.level)


def open_image(path):
	a = ScopedTimer("open_image")
	myprint("open image = " + path)
	im = Image.open(path)
	width, height = im.size
	myprint("width = " + str(width) + " height = " + str(height),1)
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	#btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	
	tmpTemplate = numpy.array(btnpixeldata, dtype=numpy.uint8)
	tmpTemplate = tmpTemplate.reshape(height, width, len(tmpTemplate[0]))
		
	return tmpTemplate, width, height
	
def toPixIndex(coord, w):
	a = ScopedTimer("toPixIndex", 0)
	if coord[0] >= w or coord[0] < 0 or coord[1] < 0:
		return -1
	return (coord[1] * w) + coord[0]
	
def toXYCoord(pixIndex, w):
	a = ScopedTimer("toXYCoord", 0)
	y = int(pixIndex / w)
	floaty = pixIndex / w
	fraction = floaty - y
	timew = fraction * w
	x = int((((pixIndex / w) - y) * w) + 0.5)
	return [x, y]

def search_all_image(path, x=0, y=0, w=-1, h=-1):
	im = Image.open(path)
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	myprint("search all " + path)
	myprint("x,y = " + str(x) + "," + str(y) + "w,h = " + str(w) + "," + str(h))
	confidence = 0.85
	coord = searchAllCoordInScreenCV(btnpixeldata, x, y, width, height, w, h, hasAlpha, confidence)
	for c in coord:
		c[0] -= int(width/2)
		c[1] -= int(height/2)
		
	myprint("coord = " + str(coord),3)
	return coord	
	
def searchAllCoordInScreenCV(pixelToFind, x, y, w, h, gwidth, gheight, hasAlpha, min_confidence=0.85):
	a = ScopedTimer("searchAllCoordInScreenCV", 1)
	if gwidth == -1 or (gwidth+x) > gScreenWidth:
		gwidth = gScreenWidth-x
	if gheight == -1 or (gheight+y) > gScreenHeight:
		gheight = gScreenHeight-y
		
	#crop image
	tmpScreen = gScreen.reshape(gScreenHeight, gScreenWidth, len(gScreen[0]))
	tmpScreen = numpy.array(tmpScreen[y:y+gheight, x:x+gwidth,0:3], dtype=numpy.uint8)
	tmpTemplate = numpy.array(pixelToFind, dtype=numpy.uint8)
	if hasAlpha:
		tmpTemplate = tmpTemplate[:,0:3]
	tmpTemplate = tmpTemplate.reshape(h,w,3)
	
	#print("(x,y) = (" + str(x) + "," + str(y) + "), width,height = " + str(gwidth) + ", " + str(gheight))
	#plt.imshow(tmpScreen)
	#<matplotlib.image.AxesImage object at 0x04123CD0>
	#plt.show()
	
	#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	res = cv2.matchTemplate(tmpScreen,tmpTemplate,cv2.TM_CCOEFF_NORMED)
	#plt.imshow(res)
	#plt.show()
	indices = numpy.argwhere(res > min_confidence)
	myprint("indices = " + str(indices),3)
	indices_centered = [ [
		int(m[1] + x + (w / 2)) + gScreenOffsetL, 
		int(m[0] + y + (h / 2)) + gScreenOffsetT]
		for m in indices]
			
	myprint("indices_centered" + str(indices_centered),3)
	return indices_centered
	
def open_image_2(path):
	src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	src_height, src_width = src.shape[:2]
	#src_hasAlpha = src.mode == "RGBA"
	
	return src, src_width, src_height
	
def run(data):
	precision = 100
	bestresult = {}
	bestresult["max_val"] = 0
	bestresult["min_val"] = 99999
	bestres = None
	indices = []
	
	src, src_width, src_height = open_image(data["source"])
	if src_width > src_height:
		src = src[31:31+696,452:452+391]
		
	src_no_alpha = src[:,:,0:3]
		
	plt.imshow(src)
	plt.show()
	
	i = 0
	
	flips = [None, 0, 1, -1]
	rotations = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]
		
	for sprite_path in data["test_sprites"]:
		sprite_orig, sprite_orig_width, sprite_orig_height = open_image(sprite_path)
		
		thisframe_res = None
		thisframe_best = 0
		
		a = ScopedTimer("Tested sprite " + sprite_path,4)
		for scale in range((int)(data["min_scale"]*precision), (int)(data["max_scale"]*precision), (int)(data["scale_step"]*precision)):
			scale_float = (float)(scale) / (float)(precision)
			myprint(scale_float)
			sprite_scaled = cv2.resize(sprite_orig, ((int)(sprite_orig_width * scale_float), (int)(sprite_orig_height * scale_float)), cv2.INTER_CUBIC)
			sprite_scaled_no_alpha = sprite_scaled[:,:,0:3]
			
			#plt.imshow(sprite_scaled)
			#plt.show()
			
			mask = sprite_scaled[:,:,3]
			ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
			#plt.imshow(mask)
			#plt.show()
			
			for f in flips:
				for r in rotations:
					sprite_scaled_fr = sprite_scaled_no_alpha
					mask_fr = mask
					if r is not None:
						sprite_scaled_fr = cv2.rotate(sprite_scaled_no_alpha, r)
						mask_fr = cv2.rotate(mask, r)
						
					if f is not None:
						sprite_scaled_fr = cv2.flip(sprite_scaled_fr,flipCode=f)
						mask_fr = cv2.flip(mask_fr, flipCode=f)
					
					res_fr = cv2.matchTemplate(src_no_alpha, sprite_scaled_fr, cv2.TM_CCORR_NORMED, mask=mask_fr)
					min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_fr)
					
					matching_indices = numpy.argwhere(res_fr > data["min_confidence"]).tolist()
					for indice in matching_indices:
						indices.append({"indice":indice, "scale":scale_float, "sprite_path":sprite_path, "w":(int)(sprite_orig_width * scale_float), "h":(int)(sprite_orig_height * scale_float), "rot":r, "flip":f})
					
					if max_val > bestresult["max_val"]:
						bestresult["max_val"] = max_val
						bestresult["min_val"] = min_val
						bestresult["indice"] = max_loc
						bestresult["min_loc"] = min_loc
						bestresult["scale"] = scale_float
						bestresult["sprite_path"] = sprite_path
						bestresult["sprite_w"] = (int)(sprite_orig_width * scale_float)
						bestresult["sprite_h"] = (int)(sprite_orig_height * scale_float)
						bestresult["rotation"] = r
						bestresult["flip"] = f
						bestres = res_fr
		
		del a
		#cv2.rectangle(src, (thisframe_res["indice"][0],thisframe_res["indice"][1]), (thisframe_res["indice"][0] + thisframe_res["sprite_w"], thisframe_res["indice"][1] + thisframe_res["sprite_h"]), i)
		i += 1
		#plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
		#plt.show()
			
	myprint("BEST : " + str(bestresult),4)
	myprint("indices : " + str(indices),5)
	src_result = src
	
	plt.imshow(bestres)
	plt.show()
	
	for indice_data in indices:		
		cv2.rectangle(src, (indice_data["indice"][1],indice_data["indice"][0]), (indice_data["indice"][1] + indice_data["w"], indice_data["indice"][0] + indice_data["h"]), 3)
	plt.imshow(cv2.cvtColor(src_result, cv2.COLOR_BGR2RGB))
	plt.show()
	
	#cv2.rectangle(src, (bestresult["min_loc"][0],bestresult["min_loc"][1]), (bestresult["min_loc"][0] + bestresult["sprite_w"], bestresult["min_loc"][1] + bestresult["sprite_h"]), 3)
	#plt.imshow(src_result)
	#plt.show()

	
if __name__ == '__main__':
	
	data = {
		"source" : os.path.join(DATA_FOLDER, "screenshots", "ss-20170524-220614.png"),
		#"test_sprites" : glob.glob( os.path.join("sprites", "chr_giant_tex", "*.png") ),
		#"test_sprites" : glob.glob( os.path.join("sprites", "chr_giant_tex", "227.png") ),
		#"test_sprites" : glob.glob( os.path.join("sprites", "chr_archer_tex", "*.png")),
		"test_sprites" : glob.glob( os.path.join("sprites", "chr_archer_tex", "105.png")),
		"min_scale" : 0.2,
		"max_scale" : 0.3,
		"scale_step" : 0.01,
		"min_confidence" : 0.96
	}
	
	run(data)
	
	#img, width, height = open_image(os.path.join("sprites", "chr_musketeer_tex", "141.png"))
	#sprite_scaled = cv2.resize(img, ((int)(width * 0.5), (int)(height * 0.3)))
	#plt.imshow(sprite_scaled)
	#plt.show()
	
	
	
		
	
	