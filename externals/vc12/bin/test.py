import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import glob
import operator
import datetime
import dateutil.relativedelta
import win32gui
import win32ui
import win32con
import win32api
import numpy
import json
import csv
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
import scipy.ndimage
import multiprocessing
import nltk
import matplotlib.pyplot as plt
import cv2
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from sklearn.externals import joblib
from time import strftime
from time import sleep
from PIL import Image
#from sklearn import svm
#from sklearn.neural_network import MLPRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import label_ranking_average_precision_score

PRINT_LEVEL=0
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))
		
def open_image(path, data):
	im = Image.open(path)
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	#btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	
	tmpTemplate = numpy.array(btnpixeldata, dtype=numpy.uint8)
	if hasAlpha:
		tmpTemplateNoAlpha = tmpTemplate[:,0:3]
		tmpTemplateAlpha = tmpTemplate[:,0:4]
		
	data["source"] = tmpTemplate.reshape(height,width,len(tmpTemplate[0]))
	data["size"] = [width, height]
	

	#subimg = data["source"] / 255
	#plt.imshow(data["source"])
	#plt.show()
	
	
	
test = {"game_area" : {
				"top":31,
				"left":452,
				"width":391,
				"height":696
			}}
def generate_smaple_pos():
	bgfiles = glob.glob(os.path.join("negative", "*.png"))
	for file in bgfiles:
		refim = cv2.imread(file)
		cv2.imwrite(file + "bleh.png", refim[31:31+696,452:452+391])
	
def shell(cmd):
	myprint("---------------------------------",3)
	myprint("OS : " + cmd,4)
	myprint("---------------------------------",3)
	return os.system(cmd)
	
def generate_all_lava_pup():
	#opencv_createsamples -img ./positive/images/000.png -bg ./negative/neg_desc.txt -info %1 -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5
	str = "opencv_createsamples -img {image} -bg ./negative/neg_desc.txt -info ./info/{output} -maxxangle 0.0 -maxyangle 0.0 -maxzangle 0.0 -w 10 -h 10"
	images = glob.glob(os.path.join("positive", "images", "*.png"))
	base_output = "list{count}.lst"
	count = 0
	for image in images:
		output = base_output.format(count=count)
		cmd = str.format(image=image, output=output)
		shell(cmd)
		count += 1
		
def collapse_list():
	files = glob.glob( os.path.join("info", "*.lst") )
	with open(os.path.join("info", "all.lst"), 'w') as outfile:
		for file in files:
			with open(file) as infile:
				outfile.write(infile.read())
	
if __name__ == '__main__':
	#generate_smaple_pos()
	generate_all_lava_pup()
	collapse_list()
	sys.exit()
	#im = "test2.jpg"
	#im = os.path.join("positive", "Screenshot_20171223-182225.jpg")
	#im = os.path.join("positive", "Screenshot_20171223-182324.jpg")
	im = os.path.join("positive", "Screenshot_20171223-182326.jpg")
	classifierpath = "data\\cascade.xml"
	data = {}
	
	open_image(im, data)
	gray = data["source"]
	gray = cv2.cvtColor(data["source"], cv2.COLOR_BGR2GRAY)
	clf = cv2.CascadeClassifier(classifierpath)
	rects = clf.detectMultiScale(gray)
	
	print(rects)
	
	for rect in rects:
		cv2.rectangle(gray, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 3)
		
	subimg = gray / 255
	plt.imshow(subimg)
	plt.show()