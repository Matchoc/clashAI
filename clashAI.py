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

EXTERNAL_FOLDER = "externals"
SAMPLE_TRAINER = os.path.join(EXTERNAL_FOLDER, "vc12", "bin", "opencv_createsamples.exe")
CASCADE_TRAINER = os.path.join(EXTERNAL_FOLDER, "vc12", "bin", "opencv_traincascade.exe")
DATA_FOLDER = "data"
RED = 2
GREEN = 1
BLUE = 0
MAX_COLOR_DIFF = 30 # 0 to 255
PRINT_LEVEL=3
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))
		
# =============================================================================
# WINAPI SEQUENCE RUN	
def moveMouse(x,y):
	win32api.SetCursorPos((x,y))	

def click(x,y):
	myprint("Click Screen at " + str((x,y)),2)
	win32api.SetCursorPos((x,y))
	sleep(.5)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
	sleep(.2)
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
gScreenOffsetT = 0
gScreenOffsetL = 0
gScreenWidth = 0
gScreenHeight = 0
gScreenData = {}
def updateScreen(hwnd = None, wait_focus=True):
	a = ScopedTimer("updateScreen")
	global gScreen
	global gScreenAlpha
	global gScreenOffsetT
	global gScreenOffsetL
	global gScreenWidth
	global gScreenHeight
	
	global gScreenNumpy
	global gScreenAlphaNumpy
	global gScreenData
	
	if not hwnd:
		hwnd=win32gui.GetDesktopWindow()
	l,t,r,b=win32gui.GetWindowRect(hwnd)
	gScreenOffsetT = t
	gScreenOffsetL = l
	h=b-t
	w=r-l
	gScreenWidth = w
	gScreenHeight = h
	if "hDC" not in gScreenData or gScreenData["hDC"] is None:
		hDC = win32gui.GetWindowDC(hwnd)
		myDC=win32ui.CreateDCFromHandle(hDC)
		newDC=myDC.CreateCompatibleDC()
		myBitMap = win32ui.CreateBitmap()
		myBitMap.CreateCompatibleBitmap(myDC, w, h)
		gScreenData["hDC"] = hDC
		gScreenData["myDC"] = myDC
		gScreenData["newDC"] = newDC
		gScreenData["myBitMap"] = myBitMap
	else:
		hDC = gScreenData["hDC"]
		myDC = gScreenData["myDC"]
		newDC = gScreenData["newDC"]
		myBitMap = gScreenData["myBitMap"]

	newDC.SelectObject(myBitMap)

	if wait_focus:
		win32gui.SetForegroundWindow(hwnd)
		sleep(.2) #lame way to allow screen to draw before taking shot
		
	newDC.BitBlt((0,0),(w, h) , myDC, (0,0), win32con.SRCCOPY)
	myBitMap.Paint(newDC)
	asTuple = myBitMap.GetBitmapBits(False)
	
	# transform asTuple into modifiable list
	gScreen = numpy.array(asTuple)
	gScreen = numpy.where(gScreen < 0, gScreen + 2**8, gScreen)
	gScreen = gScreen.reshape(gScreenHeight * gScreenWidth, 4)
	gScreenAlpha = gScreen
	
	#win32gui.ReleaseDC(win32gui.GetDesktopWindow(), hDC)
	
	myprint("screenWidth : " + str(gScreenWidth) + ", screenHeight : " + str(gScreenHeight) + ", offsetL : " + str(gScreenOffsetL)  + ", offsetT : " + str(gScreenOffsetT) + ", ndim : " + str(gScreen.ndim))

def gScreenToNumpy():
	global gScreenNumpy
	gScreenNumpy = numpy.array(gScreen)
	gScreenNumpy = gScreenNumpy.reshape(gScreenHeight, gScreenWidth, 3)
	gScreenNumpy = gScreenNumpy / 255.0
		
def takeScreenshot(hwnd = None, subfolder=None):
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
	if subfolder is not None:
		pathbmp = os.path.join(pathbmp, subfolder)
	if not os.path.isdir(pathbmp):
		os.makedirs(pathbmp)
	
	timestr = strftime("%Y%m%d-%H%M%S")
	pathbmp = os.path.join(pathbmp, "ss-" + timestr + ".png")
	
	#couldn't find another easy way to convert to png
	myBitMap.SaveBitmapFile(newDC,pathbmp)
	
# =============================================================================
# UTIL METHOD
class ScopedTimer:
	def __init__(self, name, level=3):
		self.starttime = datetime.datetime.now()
		self.name = name
		self.level = level
		
	def __del__(self):
		delta = datetime.datetime.now() - self.starttime
		myprint(str(self.name) + " : " + str(delta),self.level)

def color_diff(c1, c2):
	return abs(c1[RED] - c2[RED]) + abs(c1[GREEN] - c2[GREEN]) + abs(c1[BLUE] - c2[BLUE])
		
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
# CLUSTERING ALGO

def collectSurroundingData(pixIndex, collection, binaryList, board_size, matchAllColor = False):
	indexes = set()
	indexes.add(pixIndex)
	clusterinfo = {}
	newCluster = set()
	while len(indexes) > 0:
		index = indexes.pop()
		if not isIndexInList(index, collection):
			newCluster.add(index)
			coord = toXYCoord(index, board_size[0])
			coordu = [coord[0], coord[1] - 1]
			coordd = [coord[0], coord[1] + 1]
			coordr = [coord[0] + 1, coord[1]]
			coordl = [coord[0] - 1, coord[1]]
			indexu = toPixIndex(coordu, board_size[0])
			indexd = toPixIndex(coordd, board_size[0])
			indexr = toPixIndex(coordr, board_size[0])
			indexl = toPixIndex(coordl, board_size[0])
			if isIndexElement(indexu, binaryList) and not indexu in newCluster and (matchAllColor == False or isMatchAllColors(binaryList, index, indexu)):
				indexes.add(indexu)
			if isIndexElement(indexd, binaryList) and not indexd in newCluster and (matchAllColor == False or isMatchAllColors(binaryList, index, indexd)):
				indexes.add(indexd)
			if isIndexElement(indexr, binaryList) and not indexr in newCluster and (matchAllColor == False or isMatchAllColors(binaryList, index, indexr)):
				indexes.add(indexr)
			if isIndexElement(indexl, binaryList) and not indexl in newCluster and (matchAllColor == False or isMatchAllColors(binaryList, index, indexl)):
				indexes.add(indexl)

	minClusterSize = 5
	if len(newCluster) > minClusterSize:
		minX = -1
		minY = -1
		for index in newCluster:
			coord = toXYCoord(index, board_size[0])
			if minX < 0 or minX > coord[0]:
				minX = coord[0]
				minY = coord[1]
		perim = calculatePerimeter(newCluster, [minX, minY], board_size, False)
		clustercoord = clusterIndexToClusterCoord(newCluster, board_size)
		clusterinfo["clusterIndexes"] = newCluster
		clusterinfo["clusterPerimeter"] = perim
		clusterinfo["clusterCoord"] = clustercoord
		collection.append(clusterinfo)

def isMatchAllColors(binaryList, curIndex, newIndex):
	return binaryList[curIndex][RED] == binaryList[newIndex][RED] and binaryList[curIndex][GREEN] == binaryList[newIndex][GREEN] and binaryList[curIndex][BLUE] == binaryList[newIndex][BLUE]
		
def isIndexElement(index, binaryList):
	if index < 0 or index >= len(binaryList) or numpy.sum(binaryList[index]) <= 5:
		return False
	return True
			
def isIndexInList(index, listOfList):
	for sublist in listOfList:
		if index in sublist["clusterIndexes"]:
			return True
			
	return False
	
def collectClusters(data):
	a = ScopedTimer("collectClusters")
	myprint("Collect color clusters")
	data["frame_data"]["clusters"] = []
	start = [0,0]
	end = data["frame_data"]["arena_diff_size"]
	board_size = data["frame_data"]["arena_diff_size"]
	for y in range(start[1], end[1]):
		for x in range(start[0], end[0]):
			index = toPixIndex([x,y], board_size[0])
			if numpy.sum(data["frame_data"]["arena_diff"][index]) > 5 and not isIndexInList(index, data["frame_data"]["clusters"]):
				collectSurroundingData(index, data["frame_data"]["clusters"], data["frame_data"]["arena_diff"], board_size)
				
	myprint(str(data["frame_data"]["clusters"]),2)

def clusterIndexToClusterCoord(cluster, board_size):
	clustercoord = set()
	for index in cluster:
		clustercoord.add(tuple(toXYCoord(index, board_size[0])))
		
	return clustercoord
	
def calculatePerimeter(cluster, startCoord, board_size, verbose):
	perimeter = set()
	perimeter.add(tuple(startCoord))
	current = startCoord
	dirs = numpy.array([[0,-1], [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1]])
	backtrace = numpy.array([5, 6, 0, 0, 2, 2, 4, 4])
	clustercoord = clusterIndexToClusterCoord(cluster, board_size)
	i = 0
	start = 0
	# have to move in from an empty direction or the algorithm fails
	for x in range(len(dirs)):
		move = (start + x) % len(dirs)
		inspect = current + dirs[move]
		inspecttuple = tuple(inspect)
		if not inspecttuple in clustercoord:
			start = x
			break

	# this algo has a weakness where it will stop early.
	# the easy solution is to loop twice.
	while not numpy.array_equal(current, startCoord) or i < 10:
		if numpy.array_equal(current, startCoord):
			i += 1
		for x in range(len(dirs)):
			move = (start + x) % len(dirs)
			inspect = current + dirs[move]
			inspecttuple = tuple(inspect)
			if inspecttuple in clustercoord:
				if inspecttuple not in perimeter:
					perimeter.add(inspecttuple)
				current = inspect
				start = backtrace[move] # backtrace (http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html)
				break
	return perimeter
	
def count_pixel_per_side(data):
	a = ScopedTimer("count_pixel_per_side", 1)
	diff_img = numpy.array(data["frame_data"]["arena_diff"], dtype=numpy.uint8)
	diff_size = data["frame_data"]["arena_diff_size"]
	start_top_left = [30,63]
	end_bottom_right = [360,528]
	left_pix_count = 0
	right_pix_count = 0
	index = 0
	
	diff_img_xy = diff_img.reshape(diff_size[1], diff_size[0], len(diff_img[0]))
	if end_bottom_right[0] > diff_size[0]:
		end_bottom_right[0] = diff_size[0]
	if end_bottom_right[1] > diff_size[1]:
		end_bottom_right[1] = diff_size[1]
		
	diff_img_xy = diff_img_xy[start_top_left[1]:end_bottom_right[1],start_top_left[0]:end_bottom_right[0],0:3]
	gray_image = cv2.cvtColor(diff_img_xy, cv2.COLOR_BGR2GRAY)
	
	half_size = (end_bottom_right[0]-start_top_left[0]) / 2
	diff_img_left = gray_image[:,0:half_size]
	diff_img_right = gray_image[:,half_size:]
	
	left_pix_count = cv2.countNonZero(diff_img_left)
	right_pix_count = cv2.countNonZero(diff_img_right)
	
	#myprint(diff_img_left)
	#t = diff_img_left / 255
	#t = t.reshape(height, width, 4)
	#plt.imshow(t)
	#<matplotlib.image.AxesImage object at 0x04123CD0>
	#plt.show()
	
	#myprint(diff_img_right)
	#t = diff_img_right / 255
	#plt.imshow(t)
	#plt.show()
	
	#for pix in diff_img:
	#	coord = toXYCoord(index, diff_size[0])
	#	#myprint("coord : " + str(coord) + " pix : " + str(pix))
	#	index += 1
	#	if numpy.sum(pix[0:3]) > 5 and coord[0] > start_top_left[0] and coord[0] < end_bottom_right[0] and coord[1] > start_top_left[1] and coord[1] < end_bottom_right[1]:
	#		if coord[0] < diff_size[0] / 2:
	#			left_pix_count += 1
	#		else:
	#			right_pix_count += 1
				
	data["frame_data"]["left_count"] = left_pix_count
	data["frame_data"]["right_count"] = right_pix_count
	myprint("left count = " + str(left_pix_count) + ", right count = " + str(right_pix_count),2)
	
	
# =============================================================================
# GAME LOGIC
def get_current_screen_name(data):
	homescreen_coord = data["button_correct_coords"]["homescreen"]
	homescreen_color = data["screen_colors"]["homescreen"]
	battlescreen_coord = data["button_correct_coords"]["battlescreen"]
	battlescreen_color = data["screen_colors"]["battlescreen"]
	victoryscreen_coord = data["button_correct_coords"]["victoryscreen"]
	victoryscreen_color = data["screen_colors"]["victoryscreen"]
	changescreen_coord = data["button_correct_coords"]["changescreen"]
	changescreen_color = data["screen_colors"]["changescreen"]
	limitscreen_coord = data["button_correct_coords"]["limitscreen"]
	limitscreen_color = data["screen_colors"]["limitscreen"]
	
	homescreen_index = toPixIndex(homescreen_coord, gScreenWidth)
	battlescreen_index = toPixIndex(battlescreen_coord, gScreenWidth)
	victoryscreen_index = toPixIndex(victoryscreen_coord, gScreenWidth)
	changescreen_index = toPixIndex(changescreen_coord, gScreenWidth)
	limitscreen_index = toPixIndex(limitscreen_coord, gScreenWidth)
	
	screen_home_val = gScreen[homescreen_index]
	screen_battle_val = gScreen[battlescreen_index]
	screen_victory_val = gScreen[victoryscreen_index]
	screen_change_val = gScreen[changescreen_index]
	screen_limit_val = gScreen[limitscreen_index]
	
	myprint("color at home ({x},{y}) : {home}, at battle ({x2},{y2}) : {battle}, at victory ({x3},{y3}) : {victory}, at change ({x4},{y4}) : {change}".format(
		x=homescreen_coord[0], y=homescreen_coord[1], home=screen_home_val, x2=battlescreen_coord[0], y2=battlescreen_coord[1],
		battle=screen_battle_val, x3=victoryscreen_coord[0], y3=victoryscreen_coord[1], victory=screen_victory_val,
		x4=changescreen_coord[0], y4=changescreen_coord[1], change=screen_change_val
	), 1)
	
	diffhome = color_diff(screen_home_val, homescreen_color)
	diffbattle = color_diff(screen_battle_val, battlescreen_color)
	diffvictory = color_diff(screen_victory_val, victoryscreen_color)
	diffchange = color_diff(screen_change_val, changescreen_color)
	difflimit = color_diff(screen_limit_val, limitscreen_color)
	
	if diffhome < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "homescreen"
		return "homescreen"
	elif diffbattle < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "battlescreen"
		return "battlescreen"
	elif diffvictory < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "victoryscreen"
		return "victoryscreen"
	elif diffchange < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "changescreen"
		return "changescreen"
	elif difflimit < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "limitscreen"
		return "limitscreen"
	else:
		myprint("Error: Could not identify current screen !", 5)
		return None
	
def searchCoordInScreenCV(pixelToFind, x, y, w, h, gwidth, gheight, hasAlpha, min_confidence=0.85):
	a = ScopedTimer("searchCoordInScreenCV", 1)
	if gwidth == -1 or (gwidth+x) > gScreenWidth:
		gwidth = gScreenWidth-x
	if gheight == -1 or (gheight+y) > gScreenHeight:
		gheight = gScreenHeight-y
		
	#crop image
	myprint("gScreen dim = " + str(gScreen.ndim),1)
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
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	myprint("min_val : " + str(min_val) + ", max_val : " + str(max_val) + ", min_loc : " + str(min_loc) + ", max_loc : " + str(max_loc))
	
	# arbitrary cutoff
	if max_val < min_confidence:
		return None
	
	max_loc = list(max_loc)
	max_loc[0] = int(max_loc[0] + x + (w / 2)) + gScreenOffsetL
	max_loc[1] = int(max_loc[1] + y + (h / 2)) + gScreenOffsetT
	
	#plt.imshow(res,cmap = 'gray')
	#plt.show()
	
	return list(max_loc)
	
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
	#### ITERATE THROUGH ALL VALUES > CONFIDENCE !!!!
	
	
	
	#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	#myprint("min_val : " + str(min_val) + ", max_val : " + str(max_val) + ", min_loc : " + str(min_loc) + ", max_loc : " + str(max_loc))
	
	# arbitrary cutoff
	#if max_val < min_confidence:
	#	return None
	
	#max_loc = list(max_loc)
	#max_loc[0] = int(max_loc[0] + x + (w / 2)) + gScreenOffsetL
	#max_loc[1] = int(max_loc[1] + y + (h / 2)) + gScreenOffsetT
	
	#plt.imshow(res,cmap = 'gray')
	#plt.show()
	
	#return list(max_loc)
	

def searchCoordInScreen(pixelToFind, x, y, w, h, gwidth, gheight, hasAlpha):
	a = ScopedTimer("searchCoordInScreen", 1)
	#startindex = toPixIndex((x, y), gScreenWidth)
	#endindex = toPixIndex((x + w, y + h), gScreenWidth)
	if gwidth == -1 or (gwidth+x) > gScreenWidth:
		gwidth = gScreenWidth-x
	if gheight == -1 or (gheight+y) > gScreenHeight:
		gheight = gScreenHeight-y
	for refy in range(y, y + gheight):
		for refx in range(x, x + gwidth):
			pixIndex = toPixIndex((refx, refy), gScreenWidth)
			pix = gScreen[pixIndex]
			diff = color_diff(pix, pixelToFind[0])
			if diff <= MAX_COLOR_DIFF:
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
					screenline = screenline.tolist()
					
					for i in range(len(screenline)):
						diff = color_diff(subimgline[i], screenline[i])
						if abs(diff) > MAX_COLOR_DIFF:
							match = False
							break

					row += 1
				if match == True:
					coord = toXYCoord(pixIndex, gScreenWidth)
					coord[0] += int(w / 2) + gScreenOffsetL
					coord[1] += int(h / 2) + gScreenOffsetT
					return coord
				
	myprint("Image NOT FOUND", 2)
	return None

def convert_RGB_to_BGR(img):
	returnList = [[x[2], x[1], x[0], x[3]] for x in img]
	return returnList
	
def board_coord_to_mousepos(data, a, b):
	square_dim_x = data["drop_area"]["width"] / (data["grid_size"][0]-1)
	square_dim_y = data["drop_area"]["height"] / (data["grid_size"][1]-1)
	pos_x = data["drop_area_abs"]["left"] + (a * square_dim_x)
	pos_y = data["drop_area_abs"]["top"] + (b * square_dim_y)
	
	myprint("square_dim_x : {dimx}, square_dim_y : {dimy}, a {va}, b {vb}, abs left {aleft}, abs top {atop}, finalx {fx}, finaly {fy}".format(
		dimx=square_dim_x, dimy=square_dim_y, va=a, vb=b, aleft=data["drop_area_abs"]["left"], atop=data["drop_area_abs"]["top"], fx=pos_x, fy=pos_y))
	return int(pos_x), int(pos_y)
	
def search_image(path, x=0, y=0, w=-1, h=-1):
	im = Image.open(path)
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	myprint("search_image " + path)
	myprint("x,y = " + str(x) + "," + str(y) + "w,h = " + str(w) + "," + str(h))
	confidence = 0.85
	if "settingbtn" in path:
		confidence = 0.70
	coord = searchCoordInScreenCV(btnpixeldata, x, y, width, height, w, h, hasAlpha, confidence)
	if coord is not None:
		coord[0] -= int(width/2)
		coord[1] -= int(height/2)
	return coord
	
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
	
def calculate_offset_from_appname_ref(data):
	coord = search_image(data["ref_img"]["settingbtn"])
	myprint("coord : " + str(coord),3)
	data["appname_key"] = "settingbtn"
	data["world_ref"] = coord
	
def calculate_absolute_button_pos(data):
	data["button_abs_coords"] = {}
	data["drop_area_abs"] = {}
	appname_abs_offset_x = data["world_ref"][0] - data["button_coords"]["settingbtn"][0]
	appname_abs_offset_y = data["world_ref"][1] - data["button_coords"]["settingbtn"][1]
	for button_name in data["button_coords"]:
		world_pos_x = data["button_coords"][button_name][0] + appname_abs_offset_x
		world_pos_y = data["button_coords"][button_name][1] + appname_abs_offset_y
		data["button_abs_coords"][button_name] = (int(world_pos_x), int(world_pos_y))
		
	world_pos_x = data["drop_area"]["left"] + appname_abs_offset_x
	world_pos_y = data["drop_area"]["top"] + appname_abs_offset_y
	data["drop_area_abs"]["left"] = int(world_pos_x)
	data["drop_area_abs"]["top"] = int(world_pos_y)
	
def calculate_corrected_button_pos(data):
	data["button_correct_coords"] = {}
	for button_name in data["button_abs_coords"]:
		corrected_coord = (data["button_abs_coords"][button_name][0] - gScreenOffsetL, data["button_abs_coords"][button_name][1] - gScreenOffsetT)
		data["button_correct_coords"][button_name] = corrected_coord
		
def calculate_current_energy(data):
	a = ScopedTimer("calculate_current_energy", 1)
	data["frame_data"]["current_energy"] = 0
	energy_color = data["screen_colors"]["energybar"]
	energy_color_high = data["screen_colors"]["energybar_high"]
	for i in range(1,12):
		coord_name = "energy" + str(i)
		if coord_name not in data["button_correct_coords"]:
			break
		coord = data["button_correct_coords"][coord_name]
		coord_index = toPixIndex(coord, gScreenWidth)
		coord_val = gScreen[coord_index]
		myprint(coord_name + " : coord(" + str(coord[0]) + "," + str(coord[1]) + "), coord_val : " + str(coord_val))
		if coord_val[RED] <= 128:
			break
	
	data["frame_data"]["current_energy"] = i-1
	myprint("current energy : " + str(data["frame_data"]["current_energy"]),3)
	
def calculate_current_cards_in_hand(data):
	a = ScopedTimer("calculate_current_cards_in_hand")
	if "hand" not in data["frame_data"]:
		data["frame_data"]["hand"] = {
			"card0" : "",
			"card1" : "",
			"card2" : "",
			"card3" : ""
		}
	myprint("corrected_button_coord = " + str(data["button_abs_coords"]))
	
	startsearch = data["button_correct_coords"]["deckstarcorner"]
	width = data["button_correct_coords"]["card3"][0] + 80 - startsearch[0]
	height = data["button_correct_coords"]["card3"][1] + 80 - startsearch[1]
	
	myprint("startsearch = " + str(startsearch) + ", width,height = " + str(width) + "," + str(height))
	
	for card in data["ref_img"]["cards"]:
		coord = search_image(data["ref_img"]["cards"][card], startsearch[0], startsearch[1], width, height)
		if coord is not None:
			if coord[0] <= data["button_abs_coords"]["card0"][0]:
				data["frame_data"]["hand"]["card0"] = card
			elif coord[0] <= data["button_abs_coords"]["card1"][0]:
				data["frame_data"]["hand"]["card1"] = card
			elif coord[0] <= data["button_abs_coords"]["card2"][0]:
				data["frame_data"]["hand"]["card2"] = card
			elif coord[0] <= data["button_abs_coords"]["card3"][0]:
				data["frame_data"]["hand"]["card3"] = card
			else:
				myprint("ERROR : Invalid coord, card found at : " + str(coord) + " for card " + card, 3)
	
	# number of known card. If we don't have the mana for the card yet I probably won't be able to ID it.
	total = len([cardid for cardid in data["frame_data"]["hand"] if data["frame_data"]["hand"][cardid] is not "" and data["frame_data"]["hand"][cardid] is not None])
	return total
		
def get_arena_ref(data):
	arena_right = data["button_correct_coords"]["arena_bottom_right"][0]
	arena_bottom = data["button_correct_coords"]["arena_bottom_right"][1]
	arena_offset_x = data["button_correct_coords"]["arena_top_left"][0]
	arena_offset_y = data["button_correct_coords"]["arena_top_left"][1]
	width = arena_right - arena_offset_x
	height = arena_bottom - arena_offset_y
	
	# extract the arena picture from gScreen
	myprint("screen width " + str(gScreenWidth) + " height " + str(gScreenHeight) + " len(gscreen) " + str(len(gScreen)))
	arena_pic = gScreen.reshape(gScreenHeight, gScreenWidth, 4)
	
	arena_pic = arena_pic[arena_offset_y:arena_offset_y + height, arena_offset_x:arena_offset_x + width]
	myprint("width " + str(width) + " height " + str(height) + " arena_offset_x " + str(arena_offset_x) + " arena_offset_y " + str(arena_offset_y))
	arena_pic = arena_pic.reshape(width * height, 4)
	
	data["frame_data"]["arena_img"] = arena_pic
	data["frame_data"]["arena_diff_size"] = (width, height)
	
	#t = arena_pic / 255
	#t = t.reshape(height, width, 4)
	#plt.imshow(t)
	#<matplotlib.image.AxesImage object at 0x04123CD0>
	#plt.show()
	
def calculate_arena_diff(data):
	a = ScopedTimer("calculate_arena_diff", 1)
	cur_arena = data["current_arena"]
	
	if "arena_img" not in data["frame_data"] or data["frame_data"]["arena_img"] is None:
		get_arena_ref(data)
		
	btnpixeldata = data["frame_data"]["arena_img"]
	width, height = data["frame_data"]["arena_diff_size"]
	arena_offset_x = data["button_correct_coords"]["arena_top_left"][0]
	arena_offset_y = data["button_correct_coords"]["arena_top_left"][1]
	
	# extract the arena picture from gScreen
	myprint("screen width " + str(gScreenWidth) + " height " + str(gScreenHeight) + " len(gscreen) " + str(len(gScreen)))
	arena_pic = gScreen.reshape(gScreenHeight, gScreenWidth, 4)
	
	arena_pic = arena_pic[arena_offset_y:arena_offset_y + height, arena_offset_x:arena_offset_x + width]
	myprint("width " + str(width) + " height " + str(height) + " arena_offset_x " + str(arena_offset_x) + " arena_offset_y " + str(arena_offset_y))
	arena_pic = arena_pic.reshape(width * height, 4)
	
	#t = arena_pic / 255
	#t = t.reshape(height, width, 4)
	#plt.imshow(t)
	#<matplotlib.image.AxesImage object at 0x04123CD0>
	#plt.show()
	
	# MAX_COLOR_DIFF * 10 to try to get rid of clouds. I think the contrast between bg and units should be big enought
	sub_img = [(p[0], p[1], p[2]) if color_diff(pref, p) > (MAX_COLOR_DIFF*5) else (0,0,0) for p, pref in zip(arena_pic,btnpixeldata)]
	data["frame_data"]["arena_diff"] = sub_img
	
	#a = numpy.array(sub_img)
	#a = a / 255
	#a = a.reshape(height, width, 3)
	#plt.imshow(a)
	#<matplotlib.image.AxesImage object at 0x04123CD0>
	#plt.show()
	
def get_card(cardname, data):
	for card in data["frame_data"]["hand"]:
		if data["frame_data"]["hand"][card] == cardname:
			return card
			
	return None
	
def play_card(cardid, board_coord, data):
	myprint("Playing card " + cardid + " to " + str(board_coord),2)
	click(*data["button_abs_coords"][cardid])
	sleep(0.1)
	# bridge
	default_x = board_coord[0]
	default_y = board_coord[1]
	default_x, default_y = board_coord_to_mousepos(data, default_x, default_y)
	click(default_x, default_y)
	sleep(0.2) # give time for card to be played or sometimes next frame happen too quickly
	if "frame_data" in data:
		data["frame_data"]["hand"][cardid] = ""
		data["frame_data"]["needHandUpdate"] = True
	
def play_dumb_strat(data):
	a = ScopedTimer("play_dumb_strat")
	right_bridge = (14, 0)
	left_bridge = (3,0)
	right_rear = (9,14)
	left_rear = (7,14)
	
	# finding cards in hand is expensive. Only do it when necessary (first update and after playing a card)
	if data["frame_data"]["needHandUpdate"] == True:
		num_card_ided = calculate_current_cards_in_hand(data)
		if num_card_ided >= 4:
			data["frame_data"]["needHandUpdate"] = False
		myprint("Updated hand ided " + str(num_card_ided),1)
	else:
		num_card_ided = 4
		
	myprint("current hand : " + str(data["frame_data"]["hand"]),3)
		
	calculate_arena_diff(data)
	count_pixel_per_side(data)
	# collectClusters(data) # too costy
	calculate_current_energy(data)
	
	giant = get_card("giant", data)
	balloon = get_card("balloon", data)
	archer = get_card("archer", data)
	minion_horde = get_card("minion_horde", data)
	zap = get_card("zap", data)
	has_more_unit_on_the_left = data["frame_data"]["left_count"] > data["frame_data"]["right_count"]
	if zap is not None and data["frame_data"]["played_giant"] == True and data["frame_data"]["current_energy"] >= 2:
		play_coord = data["frame_data"]["played_giant_coord"]
		play_coord = (play_coord[0], play_coord[1] - 6)
		play_card(zap, play_coord, data)
		return
	
	if zap is None and data["frame_data"]["played_giant"] == True and data["frame_data"]["current_energy"] >= 2:
		data["frame_data"]["played_giant"] = False
	
	if giant is not None and balloon is not None and data["frame_data"]["current_energy"] >= 9:
		if has_more_unit_on_the_left:
			play_coord = right_bridge
		else:
			play_coord = left_bridge
		play_card(giant, play_coord, data)
		sleep(0.5)
		play_card(balloon, play_coord, data)
		data["frame_data"]["played_giant"] = True
		data["frame_data"]["played_giant_coord"] = play_coord
		sleep(1.0) # make sure the card was played and that mana was updated or I get weird stuff like mana is 5 but cards are grayed out
		return
		
	#if num_card_ided >= 4 and giant is not None and balloon is None and data["frame_data"]["current_energy"] >= 6:
	#	if has_more_unit_on_the_left:
	#		play_coord = right_rear
	#	else:
	#		play_coord = left_rear
	#	play_card(giant, play_coord, data)
	#	return
		
	if num_card_ided >= 4 and giant is None and balloon is not None and minion_horde is not None and data["frame_data"]["current_energy"] >= 9:
		if has_more_unit_on_the_left:
			play_coord = right_bridge
		else:
			play_coord = left_bridge
		play_card(balloon, (play_coord[0], play_coord[1] - 1), data)
		sleep(0.5)
		play_card(minion_horde, play_coord, data)
		return
		
	# wait for the mana to play the combo, no need to defend
	if giant is not None and balloon is not None:
		return
		
	fireball = get_card("fireball", data)
	minion = get_card("minion", data)
	skelarmy = get_card("skelarmy", data)
	
	if skelarmy is not None and data["frame_data"]["left_count"] > 4000 and data["frame_data"]["current_energy"] >= 3:
		play_card(skelarmy, (3,5), data)
		return
		
	if skelarmy is not None and data["frame_data"]["right_count"] > 4000 and data["frame_data"]["current_energy"] >= 3:
		play_card(skelarmy, (14,5), data)
		return
	
	if minion_horde is not None and data["frame_data"]["left_count"] > 5500 and data["frame_data"]["current_energy"] >= 7:
		play_card(minion_horde, (3,5), data)
		return
		
	if minion_horde is not None and data["frame_data"]["right_count"] > 5500 and data["frame_data"]["current_energy"] >= 7:
		play_card(minion_horde, (14,5), data)
		return
		
	if fireball is not None and data["frame_data"]["right_count"] > 6000 and data["frame_data"]["current_energy"] >= 5:
		play_card(fireball, right_bridge, data)
		return
		
	if fireball is not None and data["frame_data"]["left_count"] > 6000 and data["frame_data"]["current_energy"] >= 5:
		play_card(fireball, left_bridge, data)
		return
		
	if fireball is not None and data["frame_data"]["current_energy"] >= 8:
		play_card(fireball, (14,-8), data)
		return
		
	if archer is not None and data["frame_data"]["current_energy"] >= 8:
		play_card(archer, (8,13), data)
		return
		
	if zap is not None and data["frame_data"]["current_energy"] >= 10:
		if has_more_unit_on_the_left:
			play_coord = left_bridge
		else:
			play_coord = right_bridge
		play_card(zap, play_coord, data)
		return
	
	cheap_card = None
	if skelarmy is not None:
		cheap_card = skelarmy
	elif minion is not None:
		cheap_card = minion
	elif archer is not None:
		cheap_card = archer
	
	if cheap_card is not None and data["frame_data"]["current_energy"] >= 9:
		play_coord = (8,5)
		play_card(cheap_card, play_coord, data)
		return
	
def stuck_reset_app(data):
	click(*data["button_abs_coords"]["close_app"])
	sleep(10.0) # take it slow, don't want to fuck everything because I tried to move too fast and the computer froze up
	click(*data["button_abs_coords"]["start_app"])
	sleep(30.0)
		
def pretend(path):
	global gScreen
	global gScreenAlpha
	global gScreenNumpy
	global gScreenAlphaNumpy
	
	im = Image.open(path)
	width, height = im.size
	myprint("width = " + str(width) + " height = " + str(height),1)
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	
	tmpTemplate = numpy.array(btnpixeldata, dtype=numpy.uint8)
	if hasAlpha:
		tmpTemplateNoAlpha = tmpTemplate[:,0:3]
		tmpTemplateAlpha = tmpTemplate[:,0:4]
	#tmpTemplateNoAlpha = tmpTemplateNoAlpha.reshape(height,width,3)
	#tmpTemplateAlpha = tmpTemplateAlpha.reshape(height,width,4)
	
	gScreenNumpy = tmpTemplateNoAlpha
	gScreenAlphaNumpy = tmpTemplateAlpha
	
	gScreen = tmpTemplateAlpha
	gScreenAlpha = tmpTemplateAlpha
	
def shell(cmd):
	myprint("---------------------------------",3)
	myprint("OS : " + cmd,4)
	myprint("---------------------------------",3)
	return os.system(cmd)
	
def train_unit_ML(data):
	unit_list_path = os.path.join(DATA_FOLDER, "training_unit_list.json")
	with open(unit_list_path, 'r') as jsonfile:
		unit_list_json = json.load(jsonfile)
	
	output_base = os.path.join(DATA_FOLDER, "training")
	for folder in unit_list_json:
		#folder = r"data\sprites\chr_giant_tex"
		output_vec = os.path.join(output_base, os.path.basename(folder) + ".bin")
		#externals\vc12\bin\opencv_createsamples.exe -vec data\sprites\chr_giant_tex\sample.bin -info data\sprites\chr_giant_tex\positive.txt -bg data\sprites\chr_giant_tex\negative.txt -num 234
		num_lines = sum(1 for line in open(os.path.join(folder, "positive.txt")))
		cmd = "{opencv} -vec {vec} -info {pos} -bg {neg} -num {count}".format(opencv=SAMPLE_TRAINER, vec=output_vec, pos=os.path.join(folder, "positive.txt"), neg=os.path.join(folder, "negative.txt"), count=num_lines)
		shell(cmd)
		
		#externals\vc12\bin\opencv_traincascade.exe -data chr_giant_tex -vec data\training\chr_giant_tex.bin -bg data\sprites\chr_giant_tex\negative.txt -numPos 234 -numNeg 500
		output_cascade = os.path.join(output_base, os.path.basename(folder))
		if not os.path.isdir(output_cascade):
			os.makedirs(output_cascade)
		
		cmd = "{opencv} -data {folder} -vec {bin} -bg {neg} -numPos {numPos} -numNeg {numNeg}".format(opencv=CASCADE_TRAINER, folder=output_cascade, bin=output_vec, neg=os.path.join(folder, "negative.txt"), numPos=num_lines, numNeg=1000)
		shell(cmd)
		
def run_all(actions, data):
	if data["use_paint"] == True:
		handle = getWindowByTitle("Paint", False)
	else:
		handle = getWindowByTitle("BlueStacks", False)
		
	if handle is None or len(handle) <= 0 or handle[0] is None:
		myprint("Could not find window !", 5)
		
	if "train_unit_ML" in actions:
		train_unit_ML(data)
	
	if "takeScreenshot_test" in actions:
		while True:
			updateScreen(handle[0])
			takeScreenshot(handle[0])
			sleep(10)
	
	if "update_screen" in actions:
		updateScreen(handle[0])
		if "pretend" in actions:
			myprint("pretend",1)
			pretend(data["init_with"])
	
	if "init" in actions:
		data["frame_data"] = {}
		calculate_offset_from_appname_ref(data)
		calculate_absolute_button_pos(data)
		calculate_corrected_button_pos(data)
		a = (data["world_ref"][0] - gScreenOffsetL, data["world_ref"][1] - gScreenOffsetT)
		
		myprint("found world ref at : " + str(a) + " with : " + data["appname_key"],2)
		myprint("corrected setting coord : " + str(data["button_correct_coords"]["settingbtn"]))
		myprint("corrected start coord : " + str(data["button_correct_coords"]["battle"]))
		myprint("corrected arena top left coord : " + str(data["button_correct_coords"]["arena_top_left"]))
		
		
	if "wait_after_init" in actions:
		sleep(8)
		
	if "test_screen_diff" in actions:
		while True:
			updateScreen(handle[0])
			calculate_arena_diff(data)
			#collectClusters(data)
			count_pixel_per_side(data)
			sleep(10)
		
	if "find_screen" in actions:
		while True:
			updateScreen(handle[0])
			cur_screen = get_current_screen_name(data)
			myprint("current screen name = " + str(cur_screen),2)
			sleep(5)
				
	if "test_play_area" in actions:
		sleep(5)
		for x in range(data["grid_size"][0]):
			for y in range(data["grid_size"][1]):
				board_x, board_y = board_coord_to_mousepos(data, x, y)
				#moveMouse(data["drop_area_abs"]["left"], data["drop_area_abs"]["top"])
				#moveMouse(data["button_abs_coords"]["card2"][0], data["button_abs_coords"]["card2"][1])
				moveMouse(board_x, board_y)
				sleep(0.2)
			sleep(2)
				
	if "test_battle_button" in actions:
		moveMouse(*data["button_abs_coords"]["settingbtn"])
		sleep(4)
		moveMouse(*data["button_abs_coords"]["battle"])
		sleep(4)
		moveMouse(*data["button_abs_coords"]["shop_side"])
				
	if "test_energy" in actions:
		while True:
			updateScreen(handle[0])
			#cur_screen = get_current_screen_name(data)
			calculate_current_energy(data)
			sleep(4)
			
	if "test_cards" in actions:
		while True:
			updateScreen(handle[0])
			calculate_current_cards_in_hand(data)
			myprint("current hand : " + str(data["frame_data"]["hand"]),3)
			sleep(5)
			
	if "test_close_start_app" in actions:
		cur_time = datetime.datetime.now()
		prev_time = cur_time
		data["frame_data"]["inactive_timer"] = 0
		while True:
			updateScreen(handle[0], False) # the first init/update screen will have set the window in the foreground. Saves 0.2ms every time we do a screenshot.
			cur_screen = get_current_screen_name(data)
			data["frame_data"]["inactive_timer"] += (cur_time - prev_time).total_seconds()
			myprint("cur_scree : " + str(cur_screen) + ", inactive_timer : " + str(data["frame_data"]["inactive_timer"]))
			if data["frame_data"]["inactive_timer"] > 5.0:
				click(*data["button_abs_coords"]["close_app"])
				sleep(3.0)
				click(*data["button_abs_coords"]["start_app"])
				
			prev_time = cur_time
			cur_time = datetime.datetime.now()
		
	if "test_find_all" in actions:
		updateScreen(handle[0])
		search_all_image(data["ref_img"]["red_level"][9])
		coord = search_image(data["ref_img"]["red_level"][9])
		myprint("single search coord = " + str(coord),3)
	
	if "play" in actions:
		max_game = 300
		num_game = 0
		wait_card = 0
		cur_time = datetime.datetime.now()
		prev_time = cur_time
		data["frame_data"]["inactive_timer"] = 0
		while num_game < max_game:
			updateScreen(handle[0], False) # the first init/update screen will have set the window in the foreground. Saves 0.2ms every time we do a screenshot.
			cur_screen = get_current_screen_name(data)
			data["frame_data"]["inactive_timer"] += (cur_time - prev_time).total_seconds()
			#calculate_current_energy(data)
			if cur_screen == "homescreen":
				click(*data["button_abs_coords"]["battle"])
				data["frame_data"]["needHandUpdate"] = True
				data["frame_data"]["arena_img"] = None
				data["frame_data"]["played_giant"] = False
				data["frame_data"]["inactive_timer"] = 0
				data["frame_data"]["hand"] = {
					"card0" : "",
					"card1" : "",
					"card2" : "",
					"card3" : ""
				}
				sleep(3)
			elif cur_screen == "victoryscreen":
				if "takeScreenshot" in actions:
					updateScreen(handle[0])
					takeScreenshot(handle[0], "matches")
				num_game += 1
				myprint("playing game " + str(num_game) + "/" + str(max_game),3)
				click(*data["button_abs_coords"]["finish"])
				sleep(3)
			elif cur_screen == "battlescreen":
				play_dumb_strat(data)
				# if we've been in the battlescreen for more than 15 min there's a serious problem
				if data["frame_data"]["inactive_timer"] > 15 * 60:
					myprint("Error ! App Stuck for {}, attempting reset".format(data["frame_data"]["inactive_timer"]),5)
					data["frame_data"]["inactive_timer"] = 0
					stuck_reset_app(data)
			elif cur_screen == "changescreen":
				click(*data["button_abs_coords"]["change_ok_btn"])
				sleep(3)
			elif cur_screen == "limitscreen":
				click(*data["button_abs_coords"]["limit_ok_btn"])
				sleep(3)
			else:
				sleep(2) # avoid refreshing too fast if anyway we don't know what screen we're in.
					
			prev_time = cur_time
			cur_time = datetime.datetime.now()
				
		
if __name__ == '__main__':
	
	#handle = getWindowByTitle("Paint", False)
	#updateScreen(handle[0])
	
	#a = gScreen
	#a = a / 255
	#a = a.reshape(gScreenHeight, gScreenWidth, 4)
	#b = a[:,0:200,:]
	
	#plt.imshow(b)
	#plt.show()
	
	#sys.exit()
	
	run_all([
			#"takeScreenshot_test",
			#"train_unit_ML",
			"update_screen",
			#"pretend",
			"init",
			#"test_find_all",
			#"wait_after_init",
			#"test_screen_diff",
			#"test_cards",
			#"find_screen",
			#"test_play_area",
			#"test_battle_button",
			#"test_energy",
			#"test_close_start_app",
			
			"play",
			
			#"takeScreenshot",
			"none" # put this here so I don't have to add , when I change list size.
		],
		{
			"use_paint" : False,
			"init_with" : os.path.join(DATA_FOLDER, "screenshots", "temp.png"),
			"current_arena": "arena_6", #could detect it eventually, for now should be ok
			"ref_img" : {
				"appname" : os.path.join(DATA_FOLDER, "ref", "appname.png"),
				"settingbtn" : os.path.join(DATA_FOLDER, "ref", "settingbtn_wide.png"),
				#"settingbtn_noside" : os.path.join(DATA_FOLDER, "ref", "settingbtn_noside.png"),
				"shop_side" : os.path.join(DATA_FOLDER, "ref", "shop_noside.png"),
				#"shop_noside" : os.path.join(DATA_FOLDER, "ref", "shop_noside.png"),
				"arena_0" : os.path.join(DATA_FOLDER, "ref", "training_arena.png"), # training arena
				"arena_7" : os.path.join(DATA_FOLDER, "ref", "royal_arena.png"), # royal arena
				"arena_6" : os.path.join(DATA_FOLDER, "ref", "workshop_arena.png"), # builder's workshop
				"cards" : {
					"skelarmy" : os.path.join(DATA_FOLDER, "ref", "cardskelarmy.png"),
					"archer" : os.path.join(DATA_FOLDER, "ref", "cardarcher.png"),
					"balloon" : os.path.join(DATA_FOLDER, "ref", "cardballoon.png"),
					"fireball" : os.path.join(DATA_FOLDER, "ref", "cardfireball.png"),
					"giant" : os.path.join(DATA_FOLDER, "ref", "cardgiant2.png"),
					"goblinspear" : os.path.join(DATA_FOLDER, "ref", "cardgoblinspear.png"),
					"minion" : os.path.join(DATA_FOLDER, "ref", "cardminion.png"),
					"valkyrie" : os.path.join(DATA_FOLDER, "ref", "cardvalkyrie.png"),
					"goblin" : os.path.join(DATA_FOLDER, "ref", "cardgoblin.png"),
					"minion_horde" : os.path.join(DATA_FOLDER, "ref", "cardminionhorde.png"),
					"zap" : os.path.join(DATA_FOLDER, "ref", "cardzap2.png")
				},
				"red_level" : {
					1 : os.path.join(DATA_FOLDER, "ref", "red-level01.png"),
					2 : os.path.join(DATA_FOLDER, "ref", "red-level02.png"),
					3 : os.path.join(DATA_FOLDER, "ref", "red-level03.png"),
					4 : os.path.join(DATA_FOLDER, "ref", "red-level04.png"),
					5 : os.path.join(DATA_FOLDER, "ref", "red-level05.png"),
					6 : os.path.join(DATA_FOLDER, "ref", "red-level06.png"),
					7 : os.path.join(DATA_FOLDER, "ref", "red-level07.png"),
					8 : os.path.join(DATA_FOLDER, "ref", "red-level08.png"),
					9 : os.path.join(DATA_FOLDER, "ref", "red-level09.png")
				}
			},
			"button_coords" : {
				"battle" : (579,475),#(650,477)
				"finish" : (650,645),
				"card0" : (572,650),
				"card1" : (648,650),
				"card2" : (721,650),
				"card3" : (796,650),
				#"appname" : (245,3),
				"settingbtn" : (802,81),
				#"settingbtn_noside" : (772,81),
				#"shop_side" : (442,668),
				#"shop_noside" : (442,668),
				"homescreen" : (584,435), #(652,453),
				"battlescreen" : (683,608), #(711,598),
				"victoryscreen" : (616,625), #(673,631),
				"changescreen" : (513,695),
				"change_ok_btn" : (644,691),
				"limitscreen" : (523,326),
				"limit_ok_btn" : (648,470),
				"energy0" : (561,706),
				"energy1" : (581,706),
				"energy2" : (599,706),
				"energy3" : (626,706),
				"energy4" : (653,706),
				"energy5" : (681,706),
				"energy6" : (707,706),
				"energy7" : (735,706),
				"energy8" : (763,706),
				"energy9" : (790,706),
				"energy10" : (816,706),
				"stacksidebar" : (29,61),
				"deckstarcorner" : (520,600),
				"arena_top_left" : (460,90),
				"arena_bottom_right" : (843,566),
				"close_app" : (312,16),
				"start_app" : (108,180)
			},
			"screen_colors" : {
				"homescreen" : [83,208,255], # color of the pixel at button_coords/homescreen (BGR)
				"battlescreen" : [79,109,141],
				"victoryscreen" : [255,187,105],
				"energybar" : [244,136,240],
				"energybar_high" : [255,191,255],
				"stacksidebar" : [68,59,60],
				"changescreen" : [83,67,52],
				"limitscreen" : [241,235,222]
			},
			"game_area" : {
				"top":31,
				"left":452,
				"width":391,
				"height":696
			},
			"drop_area" : {
				"top":352,
				"left":495,
				"width":323,
				"height":209
			},
			"grid_size" : (18,15)
		})
	
	myprint("done", 5)
