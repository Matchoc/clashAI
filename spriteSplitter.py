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
from PIL import Image

PRINT_LEVEL=5
MIN_COLOR_SUM = 130
MIN_CLUSTER_SIZE = 600 #(~25x25 pixel) 30x30 looked good but golemite are too small
CV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "externals", "vc12", "bin"))
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


def open_image(path, data):
	a = ScopedTimer("open_image")
	myprint("open image = " + path)
	im = Image.open(path)
	width, height = im.size
	myprint("width = " + str(width) + " height = " + str(height),1)
	btnpixeldata = list(im.getdata())
	hasAlpha = im.mode == "RGBA"
	#btnpixeldata = convert_RGB_to_BGR(btnpixeldata)
	
	tmpTemplate = numpy.array(btnpixeldata, dtype=numpy.uint8)
	if hasAlpha:
		tmpTemplateNoAlpha = tmpTemplate[:,0:3]
		tmpTemplateAlpha = tmpTemplate[:,0:4]
		
	data["source"] = tmpTemplateAlpha
	data["size"] = [width, height]
	
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

def collectSurroundingData(pixIndex, collection, binaryList, size):
	a = ScopedTimer("collectSurroundingData", 3)
	indexes = set()
	indexes.add(pixIndex)
	clusterinfo = {}
	newCluster = set()
	while len(indexes) > 0:
		index = indexes.pop()
		if not isIndexInList(index, collection):
			newCluster.add(index)
			coord = toXYCoord(index, size[0])
			coordu = [coord[0], coord[1] - 1]
			coordd = [coord[0], coord[1] + 1]
			coordr = [coord[0] + 1, coord[1]]
			coordl = [coord[0] - 1, coord[1]]
			indexu = toPixIndex(coordu, size[0])
			indexd = toPixIndex(coordd, size[0])
			indexr = toPixIndex(coordr, size[0])
			indexl = toPixIndex(coordl, size[0])
			if isIndexElement(indexu, binaryList) and not indexu in newCluster:
				indexes.add(indexu)
			if isIndexElement(indexd, binaryList) and not indexd in newCluster:
				indexes.add(indexd)
			if isIndexElement(indexr, binaryList) and not indexr in newCluster:
				indexes.add(indexr)
			if isIndexElement(indexl, binaryList) and not indexl in newCluster:
				indexes.add(indexl)

	#minClusterSize = MIN_CLUSTER_SIZE
	#if len(newCluster) > minClusterSize:
	myprint("new cluster = {}".format(str(newCluster)),3)
	clusterinfo["clusterIndexes"] = newCluster
	collection.append(clusterinfo)
	#else:
	#	myprint("cluster too small {}".format(len(newCluster)),3)

def isMatchAllColors(binaryList, curIndex, newIndex):
	return binaryList[curIndex][RED] == binaryList[newIndex][RED] and binaryList[curIndex][GREEN] == binaryList[newIndex][GREEN] and binaryList[curIndex][BLUE] == binaryList[newIndex][BLUE]
		
def isIndexElement(index, binaryList):
	a = ScopedTimer("isIndexElement", 1)
	if index < 0 or index >= len(binaryList) or (numpy.sum(binaryList[index]) <= MIN_COLOR_SUM):
		return False
	return True
			
def isIndexInList(index, listOfList):
	a = ScopedTimer("isIndexInList", 2)
	return any(index in l["clusterIndexes"] for l in listOfList)
	
def collectCells(data):
	a = ScopedTimer("collectCells", 3)
	myprint("Collect Cells clusters")
	data["sprites"] = []
	start = [0,0]
	end = data["size"]
	for y in range(start[1], end[1]):
		for x in range(start[0], end[0]):
			index = toPixIndex([x,y], data["size"][0])
			myprint("processing ({x},{y})".format(x=x, y=y),2)
			if numpy.sum(data["source"][index]) > MIN_COLOR_SUM and not isIndexInList(index, data["sprites"]):
				collectSurroundingData(index, data["sprites"], data["source"], data["size"])
			else:
				myprint("skipped ({x},{y})".format(x=x,y=y),2)
				
	myprint("data[sprites] len = " + str(len(data["sprites"])))

def clusterIndexToClusterCoord(cluster):
	a = ScopedTimer("clusterIndexToClusterCoord")
	clustercoord = set()
	for index in cluster:
		clustercoord.add(tuple(toXYCoord(index, BOARD_SIZE[0])))
		
	return clustercoord


def drawClusters(data):
	a = ScopedTimer("drawClusters")
	counter = 0
	for cluster in data["sprites"]:
		if len(cluster["clusterIndexes"]) < MIN_CLUSTER_SIZE:
			myprint("cluster too small {}".format(len(cluster["clusterIndexes"])), 4)
			continue
		minX = data["size"][0]
		maxX = 0
		minY = data["size"][1]
		maxY = 0
		for index in cluster["clusterIndexes"]:
			coord = toXYCoord(index, data["size"][0])
			if minX > coord[0]:
				minX = coord[0]
			if maxX < coord[0]:
				maxX = coord[0]
			if minY > coord[1]:
				minY = coord[1]
			if maxY < coord[1]:
				maxY = coord[1]
			
		img2d = data["source"].reshape(data["size"][1], data["size"][0], len(data["source"][0]))
		#subimgOrig = img2d[minY:maxY,minX:maxX,:]
		subimg = numpy.zeros(shape=(maxY-minY+1, maxX-minX+1, len(img2d[0][0])), dtype=numpy.uint8)
		for index in cluster["clusterIndexes"]:
			x, y = toXYCoord(index, data["size"][0])
			color = data["source"][index]
			subimg[y-minY][x-minX] = color
				
		if maxX - minX > 0 and maxY - minY > 0:
			saveBitmap(subimg, os.path.join(data["outputdir"], "{:03}.png".format(counter)))
		else:
			myprint("skipped {} in {} because size is invalid {}, {}, {}, {}".format(counter, data["outputdir"], minX, maxX, minY, maxY),5)
		
		counter = counter + 1
		#subimg = subimg / 255
		#plt.imshow(subimg)
		#plt.show()
		
def run(file):
	a = ScopedTimer("run")
	myprint("Processing : " + file,5)
	sys.stdout.flush()
	data = {}
	sheet_name = file[:-4]
	open_image(sheet_name + ".png", data)
	outputdir = os.path.join(sheet_name)
	if not os.path.isdir(outputdir):
		os.makedirs(outputdir)
	data["outputdir"] = outputdir
	collectCells(data)
	drawClusters(data)
	myprint("Done : " + file,5)
	sys.stdout.flush()
		
def saveBitmap(img, output):
	a = ScopedTimer("saveBitmap")
	im = Image.fromarray(img)
	im.save(output)
	
def generateTrainingDesc(sprites):
	for file in sprites:
		pos_sheet_name = file[:-4]
		pos_png = glob.glob(os.path.join(pos_sheet_name, "*.png"))
		posdesc = os.path.join(pos_sheet_name, "positive.txt")
		negdesc = os.path.join(pos_sheet_name, "negative.txt")
		
		pos_list = []
		for pos_file in pos_png:
			with Image.open(pos_file) as img:
				width, height = img.size
			stripped_filename = os.path.basename(pos_file)
			pos_list.append("{image} 1 0 0 {width} {height}".format(image=stripped_filename, width=width, height=height))
			
		neg_list = []
		for sprite in sprites:
			neg_sheet_name = sprite[:-4]
			if neg_sheet_name != pos_sheet_name:
				neg_png = glob.glob(os.path.join(neg_sheet_name, "*.png"))
				for neg_file in neg_png:
					neg_file = os.path.abspath(neg_file)
					final_path = os.path.relpath(neg_file, CV_PATH)
					neg_list.append(final_path)
		
		#with open(posdesc, 'w') as f:
		myprint(str(pos_list),5)
		numpy.savetxt(posdesc, pos_list, fmt='%s', newline='\r\n')
		numpy.savetxt(negdesc, neg_list, fmt='%s', newline='\r\n')
		#with open(negdesc, 'w') as f:
		#f.write("%s\r\n" % neg_list)

	
	
	
if __name__ == '__main__':
	#data = {}
	#sheet_name = "chr_lava_pups_tex"
	
	#open_image(os.path.join("sprites", sheet_name + ".png"), data)
	
	#outputdir = os.path.join("sprites", sheet_name)
	#if not os.path.isdir(outputdir):
	#	os.makedirs(outputdir)
	#data["outputdir"] = outputdir
	
	#collectCells(data)
	
	#drawClusters(data)
	
	#run("sprites\\chr_giant_tex.png")
	
	sprites = glob.glob(os.path.join("sprites","*.png"))
	NUM_PROC = 16
	p = multiprocessing.Pool(NUM_PROC)
	r = p.map(run, sprites)
	
	#generateTrainingDesc([r"G:\Perso\projects\clashAI\externals\vc12\bin\positive\images.png", r"G:\Perso\projects\clashAI\externals\vc12\bin\negative\images.png"])
	
	
	
		
	
	