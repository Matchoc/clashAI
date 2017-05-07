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
MAX_COLOR_DIFF = 15 # 0 to 255
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
	a = ScopedTimer("updateScreen")
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
class ScopedTimer:
	def __init__(self, name):
		self.starttime = datetime.datetime.now()
		self.name = name
		
	def __del__(self):
		delta = datetime.datetime.now() - self.starttime
		myprint(str(self.name) + " : " + str(delta),3)

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
	homescreen_coord = data["button_correct_coords"]["homescreen"]
	homescreen_color = data["screen_colors"]["homescreen"]
	battlescreen_coord = data["button_correct_coords"]["battlescreen"]
	battlescreen_color = data["screen_colors"]["battlescreen"]
	victoryscreen_coord = data["button_correct_coords"]["victoryscreen"]
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
	
	diffhome = screen_home_val[RED] - homescreen_color[RED] + screen_home_val[BLUE] - homescreen_color[BLUE] + screen_home_val[GREEN] - homescreen_color[GREEN]
	diffbattle = screen_battle_val[RED] - battlescreen_color[RED] + screen_battle_val[BLUE] - battlescreen_color[BLUE] + screen_battle_val[GREEN] - battlescreen_color[GREEN]
	diffvictory = screen_victory_val[RED] - victoryscreen_color[RED] + screen_victory_val[BLUE] - victoryscreen_color[BLUE] + screen_victory_val[GREEN] - victoryscreen_color[GREEN]
	
	if abs(diffhome) < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "homescreen"
		return "homescreen"
	elif abs(diffbattle) < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "battlescreen"
		return "battlescreen"
	elif abs(diffvictory) < MAX_COLOR_DIFF:
		data["frame_data"]["current_screen"] = "victoryscreen"
		return "victoryscreen"
	else:
		myprint("Error: Could not identify current screen !", 5)
		return None
	

def searchCoordInScreen(pixelToFind, x, y, w, h, gwidth, gheight, hasAlpha):
	a = ScopedTimer("searchCoordInScreen")
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
			diff = pix[0] - pixelToFind[0][0] + pix[1] - pixelToFind[0][1] + pix[2] - pixelToFind[0][2]
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
						diff = subimgline[i][0] - screenline[i][0] + subimgline[i][1] - screenline[i][1] + subimgline[i][2] - screenline[i][2]
						#myprint("diff " + str(diff))
						if abs(diff) > MAX_COLOR_DIFF:
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
	
def search_image(path, x=0, y=0, w=-1, h=-1):
	im = Image.open(path)
	width, height = im.size
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	myprint("search_image " + path)
	coord = searchCoordInScreen(btnpixeldata, x, y, width, height, w, h, hasAlpha)
	if coord is not None:
		coord[0] -= int(width/2)
		coord[1] -= int(height/2)
	return coord
	
def calculate_offset_from_appname_ref(data):
	coord = search_image(data["ref_img"]["settingbtn"])
	data["appname_key"] = "settingbtn"
	if coord is None:
		coord = search_image(data["ref_img"]["settingbtn_noside"])
		if coord is None:
			myprint("ERROR: Screen not found from ref : " + data["ref_img"]["settingbtn"] + " or " + data["ref_img"]["settingbtn_noside"], 3)
		data["appname_key"] = "settingbtn_noside"
		
		coord2 = search_image(data["ref_img"]["shop_noside"])
		expected_x = data["button_coords"]["settingbtn"][0] - data["button_coords"]["shop_side"][0]
		expected_y = data["button_coords"]["settingbtn"][1] - data["button_coords"]["shop_side"][1]
		actual_x = data["button_coords"]["settingbtn_noside"][0] - data["button_coords"]["shop_noside"][0]
		actual_y = data["button_coords"]["settingbtn_noside"][1] - data["button_coords"]["shop_noside"][1]
		data["world_aspect"] = (actual_x / expected_x, actual_y / expected_y)
	else:
		data["world_aspect"] = (1,1)
		
	data["world_ref"] = coord
	
def calculate_absolute_button_pos(data):
	data["button_abs_coords"] = {}
	data["drop_area_abs"] = {}
	appname_abs_offset_x = data["world_ref"][0] - data["button_coords"]["settingbtn"][0]
	appname_abs_offset_y = data["world_ref"][1] - data["button_coords"]["settingbtn"][1]
	for button_name in data["button_coords"]:
		world_pos_x = (data["button_coords"][button_name][0] + appname_abs_offset_x)# * data["world_aspect"][0]
		world_pos_y = (data["button_coords"][button_name][1] + appname_abs_offset_y)# * data["world_aspect"][1]
		data["button_abs_coords"][button_name] = (int(world_pos_x), int(world_pos_y))
		
	world_pos_x = (data["drop_area"]["left"] * data["world_aspect"][0]) + appname_abs_offset_x
	world_pos_y = (data["drop_area"]["top"] * data["world_aspect"][1]) + appname_abs_offset_y
	data["drop_area_abs"]["left"] = int(world_pos_x)
	data["drop_area_abs"]["top"] = int(world_pos_y)
	
def calculate_corrected_button_pos(data):
	data["button_correct_coords"] = {}
	for button_name in data["button_abs_coords"]:
		corrected_coord = (data["button_abs_coords"][button_name][0] - gScreenOffsetL, data["button_abs_coords"][button_name][1] - gScreenOffsetT)
		data["button_correct_coords"][button_name] = corrected_coord
		
def calculate_current_energy(data):
	data["frame_data"]["current_energy"] = 0
	energy_color = data["screen_colors"]["energybar"]
	for i in range(11):
		coord_name = "energy" + str(i)
		coord = data["button_correct_coords"][coord_name]
		coord_index = toPixIndex(coord, gScreenWidth)
		coord_val = gScreen[coord_index]
		diffcolor = energy_color[RED] - coord_val[RED] + energy_color[BLUE] - coord_val[BLUE] + energy_color[GREEN] - coord_val[GREEN]
		if abs(diffcolor) > MAX_COLOR_DIFF:
			break
	
	data["frame_data"]["current_energy"] = i-1
	
def calculate_current_cards_in_hand(data):
	a = ScopedTimer("calculate_current_cards_in_hand")
	data["frame_data"]["hand"] = {
		"card0" : "",
		"card1" : "",
		"card2" : "",
		"card3" : ""
	}
	myprint("corrected_button_coord = " + str(data["button_abs_coords"]))
	
	startsearch = data["button_correct_coords"]["deckstarcorner"]
	width = data["button_correct_coords"]["card3"][0] + 100
	height = data["button_correct_coords"]["card3"][1] + 100
	
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
		
def run_all(actions, data):
	if data["use_paint"] == True:
		handle = getWindowByTitle("Paint", False)
	else:
		handle = getWindowByTitle("BlueStacks", False)
		
	if handle is None or handle[0] is None:
		myprint("Could not find window !", 5)
	
	if "takeScreenshot" in actions:
		while True:
			updateScreen(handle[0])
			takeScreenshot(handle[0])
			sleep(10)
	
	if "update_screen" in actions:
		updateScreen(handle[0])
	
	if "init" in actions:
		data["frame_data"] = {}
		calculate_offset_from_appname_ref(data)
		calculate_absolute_button_pos(data)
		calculate_corrected_button_pos(data)
		a = (data["world_ref"][0] - gScreenOffsetL, data["world_ref"][1] - gScreenOffsetT)
		
		myprint("found world ref at : " + str(a) + " with : " + data["appname_key"] + " aspect ratio " + str(data["world_aspect"]),2)
		
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
		sleep(5)
		for x in range(15):
			for y in range(15):
				board_x, board_y = board_coord_to_mousepos(data, x, y)
				#moveMouse(data["drop_area_abs"]["left"], data["drop_area_abs"]["top"])
				#moveMouse(data["button_abs_coords"]["card2"][0], data["button_abs_coords"]["card2"][1])
				moveMouse(board_x, board_y)
				sleep(0.5)
				
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
			myprint("current energy : " + str(data["frame_data"]["current_energy"]))
			sleep(2)
			
	if "test_cards" in actions:
		while True:
			updateScreen(handle[0])
			calculate_current_cards_in_hand(data)
			myprint("current hand : " + str(data["frame_data"]["hand"]),3)
			sleep(5)
	
	if "play" in actions:
		max_game = 4
		num_game = 0
		wait_card = 0
		cur_time = datetime.datetime.now()
		prev_time = cur_time
		while num_game < max_game:
			updateScreen(handle[0])
			cur_screen = get_current_screen_name(data)
			calculate_current_energy(data)
			if cur_screen == "homescreen":
				click(*data["button_abs_coords"]["battle"])
				sleep(3)
			elif cur_screen == "victoryscreen":
				num_game += 1
				click(*data["button_abs_coords"]["finish"])
				sleep(3)
			elif cur_screen == "battlescreen":
				deltat = (cur_time - prev_time).total_seconds()
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
			#"test_cards",
			#"find_screen",
			#"test_play_area",
			#"test_battle_button",
			#"test_energy",
			#"start_battle",
			"play",
			"none" # put this here so I don't have to add , when I change list size.
		],
		{
			"use_paint" : True,
			"ref_img" : {
				"appname" : os.path.join(DATA_FOLDER, "ref", "appname.png"),
				"settingbtn" : os.path.join(DATA_FOLDER, "ref", "settingbtn.png"),
				"settingbtn_noside" : os.path.join(DATA_FOLDER, "ref", "settingbtn_noside.png"),
				"shop_side" : os.path.join(DATA_FOLDER, "ref", "shop_side.png"),
				"shop_noside" : os.path.join(DATA_FOLDER, "ref", "shop_noside.png"),
				"cards" : {
					"skelarmy" : os.path.join(DATA_FOLDER, "ref", "cardskelarmy.png"),
					"archer" : os.path.join(DATA_FOLDER, "ref", "cardarcher.png"),
					"balloon" : os.path.join(DATA_FOLDER, "ref", "cardballoon.png"),
					"fireball" : os.path.join(DATA_FOLDER, "ref", "cardfireball.png"),
					"giant" : os.path.join(DATA_FOLDER, "ref", "cardgiant.png"),
					"goblinspear" : os.path.join(DATA_FOLDER, "ref", "cardgoblinspear.png"),
					"minion" : os.path.join(DATA_FOLDER, "ref", "cardminion.png"),
					"valkyrie" : os.path.join(DATA_FOLDER, "ref", "cardvalkyrie.png"),
					"goblin" : os.path.join(DATA_FOLDER, "ref", "cardgoblin.png")
				}
			},
			"button_coords" : {
				"battle" : (677,477),
				"finish" : (676,645),
				"card0" : (600,650),
				"card1" : (675,650),
				"card2" : (750,650),
				"card3" : (825,650),
				"appname" : (245,3),
				"settingbtn" : (831,81),
				"settingbtn_noside" : (772,81),
				"shop_side" : (493,668),
				"shop_noside" : (442,668),
				"homescreen" : (683,453),
				"battlescreen" : (711,598),
				"victoryscreen" : (673,631),
				"energy0" : (588,706),
				"energy1" : (608,706),
				"energy2" : (634,706),
				"energy3" : (658,706),
				"energy4" : (687,706),
				"energy5" : (711,706),
				"energy6" : (738,706),
				"energy7" : (766,706),
				"energy8" : (794,706),
				"energy9" : (822,706),
				"energy10" : (848,706),
				"stacksidebar" : (29,61),
				"deckstarcorner" : (569,607)				
			},
			"screen_colors" : {
				"homescreen" : [83,208,255], # color of the pixel at button_coords/homescreen (GRB)
				"battlescreen" : [101,135,166],
				"victoryscreen" : [255,187,105],
				"energybar" : [244,136,240],
				"stacksidebar" : [68,59,60]
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
