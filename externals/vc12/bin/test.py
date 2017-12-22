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
		
	
if __name__ == '__main__':
	#generate_smaple_pos()
	im = "test2.jpg"
	classifierpath = "data\\cascade.xml"
	data = {}
	
	open_image(im, data)
	gray = data["source"]
	#gray = cv2.cvtColor(data["source"], cv2.COLOR_BGR2GRAY)
	clf = cv2.CascadeClassifier(classifierpath)
	rects = clf.detectMultiScale(gray)
	
	print(rects)
	
	for rect in rects:
		cv2.rectangle(gray, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 3)
		
	subimg = gray / 255
	plt.imshow(subimg)
	plt.show()