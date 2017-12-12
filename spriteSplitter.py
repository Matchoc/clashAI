import sys
import os
os.environ["path"] = os.path.dirname(sys.executable) + ";" + os.environ["path"]
import glob
import operator
import numpy
import json
import scipy.ndimage
from PIL import Image

PRINT_LEVEL=0
def myprint(msg, level=0):
	if (level >= PRINT_LEVEL):
		sys.stdout.buffer.write((str(msg) + "\n").encode('UTF-8'))

def open_image(path, data):
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
	if coord[0] >= w or coord[0] < 0 or coord[1] < 0:
		return -1
	return (coord[1] * w) + coord[0]
	
def toXYCoord(pixIndex, w):
	y = int(pixIndex / w)
	floaty = pixIndex / w
	fraction = floaty - y
	timew = fraction * w
	x = int((((pixIndex / w) - y) * w) + 0.5)
	return [x, y]

def collectSurroundingData(pixIndex, collection, binaryList, size):
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

	minClusterSize = 5
	if len(newCluster) > minClusterSize:
		minX = -1
		minY = -1
		for index in newCluster:
			coord = toXYCoord(index, data["size"][0])
			if minX < 0 or minX > coord[0]:
				minX = coord[0]
				minY = coord[1]
		#perim = calculatePerimeter(newCluster, [minX, minY], False)
		#clustercoord = clusterIndexToClusterCoord(newCluster)
		clusterinfo["clusterIndexes"] = newCluster
		#clusterinfo["clusterPerimeter"] = perim
		#clusterinfo["clusterCoord"] = clustercoord
		collection.append(clusterinfo)

def isMatchAllColors(binaryList, curIndex, newIndex):
	return binaryList[curIndex][RED] == binaryList[newIndex][RED] and binaryList[curIndex][GREEN] == binaryList[newIndex][GREEN] and binaryList[curIndex][BLUE] == binaryList[newIndex][BLUE]
		
def isIndexElement(index, binaryList):
	if index < 0 or index >= len(binaryList) or (numpy.sum(binaryList[index]) <= 0):
		return False
	return True
			
def isIndexInList(index, listOfList):
	for sublist in listOfList:
		if index in sublist["clusterIndexes"]:
			return True
			
	return False
	
def collectCells(data):
	myprint("Collect Cells clusters")
	data["sprites"] = []
	start = [0,0]
	end = data["size"]
	for y in range(start[1], end[1]):
		for x in range(start[0], end[0]):
			index = toPixIndex([x,y], data["size"][0])
			if numpy.sum(data["source"][index]) > 0 and not isIndexInList(index, data["sprites"]):
				collectSurroundingData(index, data["sprites"], data["source"], data["size"])
				
	myprint("data[sprites] len = " + str(len(data["sprites"])))

def clusterIndexToClusterCoord(cluster):
	clustercoord = set()
	for index in cluster:
		clustercoord.add(tuple(toXYCoord(index, BOARD_SIZE[0])))
		
	return clustercoord


if __name__ == '__main__':
	data = {}
	sheet_name = "chr_lava_pups_tex"
	open_image(os.path.join("sprites", sheet_name + ".png"), data)
	
	outputdir = os.path.join("sprites", sheet_name)
	if not os.path.isdir(outputdir):
		os.makedirs(outputdir)
	
	collectCells(data)
	