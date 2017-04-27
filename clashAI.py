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
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from sklearn.externals import joblib
from time import strftime
from time import sleep
from PIL import Image
#from sklearn import svm
#from sklearn.neural_network import MLPRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import label_ranking_average_precision_score

DATA_FOLDER = "data"
PRINT_LEVEL=1
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))
		
# =============================================================================
# WINAPI SEQUENCE RUN	
def moveMouse(x,y):
	win32api.SetCursorPos((x,y))	

def click(x,y):
	win32api.SetCursorPos((x,y))
	sleep(.5)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
	sleep(.5)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)		

def getWindowByTitle(title_text, exact = False):
	def _window_callback(hwnd, all_windows):
		all_windows.append((hwnd, win32gui.GetWindowText(hwnd)))
	windows = []
	win32gui.EnumWindows(_window_callback, windows)
	if exact:
		return [hwnd for hwnd, title in windows if title_text == title]
	else:
		return [hwnd for hwnd, title in windows if title_text in title]
		
gScreen = []
gScreenAlpha = []
gScreenNumpy = []
gScreenAlphaNumpy = []
gScreenOffsetT = 0
gScreenOffsetL = 0
gScreenWidth = 0
gScreenHeight = 0
def updateScreen(hwnd = None):
	global gScreen
	global gScreenAlpha
	global gScreenOffsetT
	global gScreenOffsetL
	global gScreenWidth
	global gScreenHeight
	
	global gScreenNumpy
	global gScreenAlphaNumpy
	
	if not hwnd:
		hwnd=win32gui.GetDesktopWindow()
	l,t,r,b=win32gui.GetWindowRect(hwnd)
	gScreenOffsetT = t
	gScreenOffsetL = l
	h=b-t
	w=r-l
	gScreenWidth = w
	gScreenHeight = h
	hDC = win32gui.GetWindowDC(hwnd)
	myDC=win32ui.CreateDCFromHandle(hDC)
	newDC=myDC.CreateCompatibleDC()

	myBitMap = win32ui.CreateBitmap()
	myBitMap.CreateCompatibleBitmap(myDC, w, h)

	newDC.SelectObject(myBitMap)

	win32gui.SetForegroundWindow(hwnd)
	sleep(.2) #lame way to allow screen to draw before taking shot
	newDC.BitBlt((0,0),(w, h) , myDC, (0,0), win32con.SRCCOPY)
	myBitMap.Paint(newDC)
	asTuple = myBitMap.GetBitmapBits(False)
	# transform asTuple into modifiable list
	gScreen = asPILFormat(asTuple, False)
	gScreenAlpha = asPILFormat(asTuple, True)
	gScreenToNumpy()
	
	myprint("screenWidth : " + str(gScreenWidth) + ", screenHeight : " + str(gScreenHeight) + ", offsetL : " + str(gScreenOffsetL)  + ", offsetT : " + str(gScreenOffsetT))

def gScreenToNumpy():
	global gScreenNumpy
	gScreenNumpy = numpy.array(gScreen)
	gScreenNumpy = gScreenNumpy.reshape(gScreenHeight, gScreenWidth, 3)
	gScreenNumpy = gScreenNumpy / 255.0
		
def takeScreenshot(hwnd = None):
	global gScreenshotCount
	if not hwnd:
		hwnd=win32gui.GetDesktopWindow()
	l,t,r,b=win32gui.GetWindowRect(hwnd)
	gScreenOffsetT = t
	gScreenOffsetL = l
	h=b-t
	w=r-l
	gScreenWidth = w
	gScreenHeight = h
	hDC = win32gui.GetWindowDC(hwnd)
	myDC=win32ui.CreateDCFromHandle(hDC)
	newDC=myDC.CreateCompatibleDC()

	myBitMap = win32ui.CreateBitmap()
	myBitMap.CreateCompatibleBitmap(myDC, w, h)

	newDC.SelectObject(myBitMap)

	win32gui.SetForegroundWindow(hwnd)
	sleep(.2) #lame way to allow screen to draw before taking shot
	newDC.BitBlt((0,0),(w, h) , myDC, (0,0), win32con.SRCCOPY)
	myBitMap.Paint(newDC)	
	
	pathbmp = os.path.join(DATA_FOLDER, "screenshots")
	if not os.path.isdir(pathbmp):
		os.makedirs(pathbmp)
	
	timestr = strftime("%Y%m%d-%H%M%S")
	pathbmp = os.path.join(pathbmp, "ss-" + timestr + ".png")
	
	#couldn't find another easy way to convert to png
	myBitMap.SaveBitmapFile(newDC,pathbmp)
	
# =============================================================================
# UTIL METHOD
def toXYCoord(pixIndex, w):
	y = int(pixIndex / w)
	floaty = pixIndex / w
	fraction = floaty - y
	timew = fraction * w
	x = int((((pixIndex / w) - y) * w) + 0.5)
	return [x, y]

def parseUint(val):
	if val < 0:
		return val + 2**8
	else:
		return val
	
def asPILFormat(asTuple, hasAlpha):
	if hasAlpha:
		returnList = [
			tuple(
				[
					parseUint(asTuple[(x*4)+2]), 
					parseUint(asTuple[(x*4)+1]), 
					parseUint(asTuple[(x*4)]), 
					255
				]
			) for x in range(int(len(asTuple) / 4))]
	else:
		returnList = [
			tuple(
				[
					parseUint(asTuple[(x*4)+2]), 
					parseUint(asTuple[(x*4)+1]), 
					parseUint(asTuple[(x*4)])
				]
			) for x in range(int(len(asTuple) / 4))]
	return returnList
	
def toPixIndex(coord, w):
	if coord[0] >= w or coord[0] < 0 or coord[1] < 0:
		return -1
	return (coord[1] * w) + coord[0]
	
# =============================================================================
# GAME LOGIC
def searchCoordInScreen(pixelToFind, w, h, hasAlpha):
	for pixIndex in range(len(gScreen)):
		pix = gScreen[pixIndex]
		if pix[0] == pixelToFind[0][0] and pix[1] == pixelToFind[0][1] and pix[2] == pixelToFind[0][2]:
			match = True
			row = 0
			while(match and row < 1):
				coordscreen = toXYCoord(pixIndex, gScreenWidth)
				coordscreen[0] += row
				screenIndex = toPixIndex(coordscreen, gScreenWidth)
				if screenIndex > len(gScreen):
					match = False
					break;
				
				coordimg = (row, 0)
				imgIndex = toPixIndex(coordimg, w)
				
				screenline = []
				if hasAlpha:
					screenline = gScreenAlpha[screenIndex:screenIndex+w]
				else:
					screenline = gScreen[screenIndex:screenIndex+w]
				subimgline = pixelToFind[imgIndex:imgIndex+w]
				intersectpix = set(subimgline).intersection(screenline)
				#myprint("Found intersection line " + str(row) + " : " + str(intersectpix))
				if screenline != subimgline:
					match = False
				row += 1
			if match == True:
				coord = toXYCoord(pixIndex, gScreenWidth)
				coord[0] += int(w / 2) + gScreenOffsetL
				coord[1] += int(h / 2) + gScreenOffsetT
				return coord
	return None

def calculate_offset_from_appname_ref(data):
	im = Image.open(data["ref_img"]["appname"])
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	coord = searchCoordInScreen(btnpixeldata, width, height, hasAlpha)
	coord[0] -= int(width/2)
	coord[1] -= int(height/2)
	data["appname_world_ref"] = coord
	
def calculate_absolute_button_pos(data):
	data["button_abs_coords"] = {}
	appname_abs_offset_x = data["appname_world_ref"][0] - data["button_coords"]["appname"][0]
	appname_abs_offset_y = data["appname_world_ref"][1] - data["button_coords"]["appname"][1]
	for button_name in data["button_coords"]:
		world_pos_x = data["button_coords"][button_name][0] + appname_abs_offset_x
		world_pos_y = data["button_coords"][button_name][1] + appname_abs_offset_y
		data["button_abs_coords"][button_name] = (world_pos_x, world_pos_y)

def run_all(actions, data):
	if data["use_paint"] == True:
		handle = getWindowByTitle("Paint", False)
	else:
		handle = getWindowByTitle("BlueStacks", False)
	updateScreen(handle[0])
	calculate_offset_from_appname_ref(data)
	calculate_absolute_button_pos(data)
	myprint("found appname ref at : " + str(data["appname_world_ref"]),2)
	moveMouse(*data["button_abs_coords"]["card3"])
	#takeScreenshot(handle[0])
		
if __name__ == '__main__':
	
	run_all([
			"none" # put this here so I don't have to add , when I change list size.
		],
		{
			"use_paint" : True,
			"ref_img" : {
				"appname" : os.path.join(DATA_FOLDER, "ref", "appname.png")
			},
			"button_coords" : {
				"battle" : (677,477),
				"finish" : (676,645),
				"card0" : (600,650),
				"card1" : (675,650),
				"card2" : (750,650),
				"card3" : (825,650),
				"appname" : (245,3)
			},
			"game_area" : {
				"top":31,
				"left":481,
				"width":391,
				"height":696
			},
			"drop_area" : {
				"top":340,
				"left":452,
				"width":333,
				"height":210
			},
			"grid_size" : (18,15)
		})
	
	myprint("done", 5)
