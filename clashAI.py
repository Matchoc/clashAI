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
RED = 2
GREEN = 1
BLUE = 0
PRINT_LEVEL=0
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
gScreenOffsetT = 0
gScreenOffsetL = 0
gScreenWidth = 0
gScreenHeight = 0
def updateScreen(hwnd = None, wait_focus=True):
	starttime = datetime.datetime.now()
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
	if wait_focus:
		sleep(.2) #lame way to allow screen to draw before taking shot
	newDC.BitBlt((0,0),(w, h) , myDC, (0,0), win32con.SRCCOPY)
	myBitMap.Paint(newDC)
	asTuple = myBitMap.GetBitmapBits(False)
	
	# transform asTuple into modifiable list
	gScreen = numpy.array(asTuple)
	gScreen = numpy.where(gScreen < 0, gScreen + 2**8, gScreen)
	gScreen = gScreen.reshape(gScreenHeight * gScreenWidth, 4)
	gScreenAlpha = gScreen
	
	myprint("screenWidth : " + str(gScreenWidth) + ", screenHeight : " + str(gScreenHeight) + ", offsetL : " + str(gScreenOffsetL)  + ", offsetT : " + str(gScreenOffsetT))
	delta = datetime.datetime.now() - starttime
	myprint("updateScreen took " + str(delta),3)	

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
def get_current_screen_name(data):
	homescreen_coord = (data["button_abs_coords"]["homescreen"][0] - gScreenOffsetL, data["button_abs_coords"]["homescreen"][1] - gScreenOffsetT)
	homescreen_color = data["screen_colors"]["homescreen"]
	battlescreen_coord = (data["button_abs_coords"]["battlescreen"][0] - gScreenOffsetL, data["button_abs_coords"]["battlescreen"][1] - gScreenOffsetT)
	battlescreen_color = data["screen_colors"]["battlescreen"]
	victoryscreen_coord = (data["button_abs_coords"]["victoryscreen"][0] - gScreenOffsetL, data["button_abs_coords"]["victoryscreen"][1] - gScreenOffsetT)
	victoryscreen_color = data["screen_colors"]["victoryscreen"]
	
	homescreen_index = toPixIndex(homescreen_coord, gScreenWidth)
	battlescreen_index = toPixIndex(battlescreen_coord, gScreenWidth)
	victoryscreen_index = toPixIndex(victoryscreen_coord, gScreenWidth)
	
	screen_home_val = gScreen[homescreen_index]
	screen_battle_val = gScreen[battlescreen_index]
	screen_victory_val = gScreen[victoryscreen_index]
	
	myprint("color at home ({x},{y}) : {home}, at battle ({x2},{y2}) : {battle}, at victory ({x3},{y3}) : {victory}".format(
		x=homescreen_coord[0], y=homescreen_coord[1], home=screen_home_val, x2=battlescreen_coord[0], y2=battlescreen_coord[1],
		battle=screen_battle_val, x3=victoryscreen_coord[0], y3=victoryscreen_coord[1], victory=screen_victory_val
	), 2)
	
	if screen_home_val[RED] == homescreen_color[RED] and screen_home_val[BLUE] == homescreen_color[BLUE] and screen_home_val[GREEN] == homescreen_color[GREEN]:
		return "homescreen"
	elif screen_battle_val[RED] == battlescreen_color[RED] and screen_battle_val[BLUE] == battlescreen_color[BLUE] and screen_battle_val[GREEN] == battlescreen_color[GREEN]:
		return "battlescreen"
	elif screen_victory_val[RED] == victoryscreen_color[RED] and screen_victory_val[BLUE] == victoryscreen_color[BLUE] and screen_victory_val[GREEN] == victoryscreen_color[GREEN]:
		return "victoryscreen"
	else:
		myprint("Error: Could not identify current screen !", 5)
		return None
	

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
				screenline = screenline.tolist()
				
				for i in range(len(screenline)):
					if subimgline[i][0] != screenline[i][0] or subimgline[i][1] != screenline[i][1] or subimgline[i][2] != screenline[i][2]:
						match = False
						break
				
				#intersectpix = set(subimgline).intersection(screenline)
				#myprint("Found intersection line " + str(row) + " : " + str(intersectpix))
				#if screenline != subimgline:
				#	match = False
				row += 1
			if match == True:
				coord = toXYCoord(pixIndex, gScreenWidth)
				coord[0] += int(w / 2) + gScreenOffsetL
				coord[1] += int(h / 2) + gScreenOffsetT
				return coord
	return None

def convert_RGB_to_BGR(img):
	returnList = [[x[2], x[1], x[0], x[3]] for x in img]
	return returnList
	
def board_coord_to_mousepos(data, a, b):
	square_dim_x = data["drop_area"]["width"] / data["grid_size"][0]
	square_dim_y = data["drop_area"]["height"] / data["grid_size"][1]
	pos_x = data["drop_area_abs"]["left"] + (a * square_dim_x)
	pos_y = data["drop_area_abs"]["top"] + (b * square_dim_y)
	
	myprint("square_dim_x : {dimx}, square_dim_y : {dimy}, a {va}, b {vb}, abs left {aleft}, abs top {atop}, finalx {fx}, finaly {fy}".format(
		dimx=square_dim_x, dimy=square_dim_y, va=a, vb=b, aleft=data["drop_area_abs"]["left"], atop=data["drop_area_abs"]["top"], fx=pos_x, fy=pos_y))
	return int(pos_x), int(pos_y)
	
def calculate_offset_from_appname_ref(data):
	im = Image.open(data["ref_img"]["appname"])
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	coord = searchCoordInScreen(btnpixeldata, width, height, hasAlpha)
	coord[0] -= int(width/2)
	coord[1] -= int(height/2)
	data["appname_world_ref"] = coord
	
def calculate_absolute_button_pos(data):
	data["button_abs_coords"] = {}
	data["drop_area_abs"] = {}
	appname_abs_offset_x = data["appname_world_ref"][0] - data["button_coords"]["appname"][0]
	appname_abs_offset_y = data["appname_world_ref"][1] - data["button_coords"]["appname"][1]
	for button_name in data["button_coords"]:
		world_pos_x = data["button_coords"][button_name][0] + appname_abs_offset_x
		world_pos_y = data["button_coords"][button_name][1] + appname_abs_offset_y
		data["button_abs_coords"][button_name] = (world_pos_x, world_pos_y)
		
	world_pos_x = data["drop_area"]["left"] + appname_abs_offset_x
	world_pos_y = data["drop_area"]["top"] + appname_abs_offset_y
	data["drop_area_abs"]["left"] = world_pos_x
	data["drop_area_abs"]["top"] = world_pos_y
		
def run_all(actions, data):
	if data["use_paint"] == True:
		handle = getWindowByTitle("Paint", False)
	else:
		handle = getWindowByTitle("BlueStacks", False)
	
	if "takeScreenshot" in actions:
		takeScreenshot(handle[0])
	
	if "update_screen" in actions:
		updateScreen(handle[0])
	
	if "init" in actions:
		calculate_offset_from_appname_ref(data)
		calculate_absolute_button_pos(data)
		myprint("found appname ref at : " + str(data["appname_world_ref"]),2)
		
	if "find_screen" in actions:
		cur_screen = get_current_screen_name(data)
		myprint("current screen name = " + cur_screen,2)
		
		if "start_battle" in actions:
			if cur_screen == "homescreen":
				click(*data["button_abs_coords"]["battle"])
				sleep(3)
				updateScreen(handle[0])
				cur_screen = get_current_screen_name(data)
				myprint("battle should be starting, current screen : " + str(cur_screen))
				
	if "test_play_area" in actions:
		for x in range(1):
			for y in range(1):
				#board_x, board_y = board_coord_to_mousepos(data, x, y)
				moveMouse(data["drop_area_abs"]["left"], data["drop_area_abs"]["top"])
				#moveMouse(data["button_abs_coords"]["card2"][0], data["button_abs_coords"]["card2"][1])
				#moveMouse(board_x, board_y)
				sleep(2)
	
	if "play" in actions:
		max_game = 4
		num_game = 0
		wait_card = 0
		cur_time = datetime.datetime.now()
		prev_time = cur_time
		while num_game < max_game:
			updateScreen(handle[0])
			cur_screen = get_current_screen_name(data)
			if cur_screen == "homescreen":
				click(*data["button_abs_coords"]["battle"])
				sleep(3)
			elif cur_screen == "victoryscreen":
				num_game += 1
				click(*data["button_abs_coords"]["victory"])
				sleep(3)
			elif cur_screen == "battlescreen":
				deltat = (cur_time - prev_time).seconds
				myprint("deltat = {dt}, wait_card = {wt}".format(dt=deltat, wt=wait_card))
				wait_card -= deltat
				if wait_card <= 0:
					click(*data["button_abs_coords"]["card0"])
					sleep(0.1)
					# front of my right tower
					default_x = 15
					default_y = 6
					default_x, default_y = board_coord_to_mousepos(data, default_x, default_y)
					click(default_x, default_y)
					wait_card = 5.0
					
			prev_time = cur_time
			cur_time = datetime.datetime.now()
				
		
if __name__ == '__main__':
	
	run_all([
			#"takeScreenshot",
			"update_screen",
			"init",
			"find_screen",
			#"test_play_area",
			#"start_battle",
			"play",
			"none" # put this here so I don't have to add , when I change list size.
		],
		{
			"use_paint" : False,
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
				"appname" : (245,3),
				"homescreen" : (683,453),
				"battlescreen" : (711,598),
				"victoryscreen" : (673,631)
			},
			"screen_colors" : {
				"homescreen" : [83,208,255], # color of the pixel at button_coords/homescreen (GRB)
				"battlescreen" : [101,135,166],
				"victoryscreen" : [255,187,105]
			},
			"game_area" : {
				"top":31,
				"left":481,
				"width":391,
				"height":696
			},
			"drop_area" : {
				"top":348,
				"left":520,
				"width":318,
				"height":210
			},
			"grid_size" : (18,15)
		})
	
	myprint("done", 5)
