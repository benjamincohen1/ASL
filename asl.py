import cv
import cv2
from SimpleCV import *
import pygame
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import itertools
import time
def main():
	#NEURAL NET CODE#
	retrain = False
	net = buildNetwork(20, 23, 26, bias = True)
	ds = SupervisedDataSet(20, 26)
	for line in open('trainingA.txt'):
		line = line.split(",")
		ds.addSample(line[0:20], (line[20:]))

	if retrain:
		trainer = BackpropTrainer(net, ds)
		print "TRAINING"
		error = 100
		t1 = time.time()
		ep = 0
		while error > .003:
			error = trainer.train()
			ep += 1
			print error



		print "TRAINED IN " + str(time.time() - t1) +" SECONDS AND " + str(ep) +" EPOCHS"
		NetworkWriter.writeToFile(net, 'net.xml')
	else:
		net = NetworkReader.readFrom('net.xml')

	#END NN CODE##
	cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
	capture = cv.CaptureFromCAM(0)
	As = 0
	Bs = 0
	while True:
		cv.WaitKey(100)
		frame = cv.QueryFrame(capture)
		if frame != None:
			simplecvimg = Image(frame, cv2image=True).crop(100,100,540,380)


			#640x480
			green = simplecvimg.colorDistance((22,75,63)) * 2
			red = simplecvimg.colorDistance((174,57,52)) * 1.5
			blue = simplecvimg.colorDistance((17,38,91)) * 2.5
			yellow = simplecvimg.colorDistance((181,161,64)) * 2
			brown = simplecvimg.colorDistance((69,28,29)) * 3.5

			
			greenNail = simplecvimg - (green)
			redNail = simplecvimg - red
			blueNail = simplecvimg - blue
			yellowNail = simplecvimg - yellow
			brownNail = simplecvimg - brown


			greenBlobs = greenNail.findBlobs()
			redBlobs = redNail.findBlobs()
			blueBlobs = blueNail.findBlobs()
			yellowBlobs = yellowNail.findBlobs()
			brownBlobs = brownNail.findBlobs()

			l1 = DrawingLayer((simplecvimg.width, simplecvimg.height))

			l2 = DrawingLayer((simplecvimg.width, simplecvimg.height))
			newblobs = {}
			if greenBlobs:
				for b in greenBlobs:
					if b.area() < 100 or b.area() > 750:
						if b.isCircle(tolerance=.1) == False:
							greenBlobs.remove(b)
				for x in greenBlobs:
					if len(greenBlobs) == 1:
						newblobs['green'] = x.centroid()
					x.drawRect(layer=l2, color=Color.GREEN, width=2, alpha=255)

			if redBlobs:
				for b in redBlobs:
					if b.area() < 100 or b.area() > 1000:
						if b.isCircle(tolerance=.1) == False:
							redBlobs.remove(b)
				#blobs.draw(color=Color.RED, width=2)
				for x in redBlobs:
					if len(redBlobs) == 1:
						newblobs['red'] = x.centroid()
					x.drawRect(layer=l2, color=Color.RED, width=2, alpha=255)


			if yellowBlobs:
				for b in yellowBlobs:
					if b.area() < 100 or b.area() > 750:
						if b.isCircle(tolerance=.1) == False:
							yellowBlobs.remove(b)
				#blobs.draw(color=Color.RED, width=2)
				for x in yellowBlobs:
					if len(yellowBlobs) == 1:
						newblobs['yellow'] = x.centroid()
					# x.drawRect(layer=l2, color=Color.WHITE, width=2, alpha=255)
					x.drawRect(layer=l2, color=Color.ORANGE, width=2, alpha=255)


			if blueBlobs:
				for b in blueBlobs:
					if b.area() < 100 or b.area() > 500:
						if b.isCircle(tolerance=.1) == False:
							blueBlobs.remove(b)
				#blobs.draw(color=Color.RED, width=2)
				for x in blueBlobs:
					if len(blueBlobs) == 1:
						newblobs['blue'] = x.centroid()
					x.drawRect(layer=l2, color=Color.BLUE, width=2, alpha=255)

			if brownBlobs:
				for b in brownBlobs:
					if b.area() < 100 or b.area() > 500:
						if b.isCircle(tolerance=.1) == False:
							brownBlobs.remove(b)
				#blobs.draw(color=Color.RED, width=2)
				for x in brownBlobs:
					if len(brownBlobs) == 1:
						newblobs['brown'] = x.centroid()
					x.drawRect(layer=l2, color=Color.LIME, width=2, alpha=255)

			# if blobs != None:
			# 	for blob in blobs:
			# 		print [round(x,0) for x in blob.centroid()]
			print newblobs
			simplecvimg.addDrawingLayer(l2)
			simplecvimg.applyLayers()
			f = open('trainingA.txt', 'a')
			# print i
			if 'red' in newblobs and 'blue' in newblobs and 'yellow'  in newblobs and 'brown' in newblobs and 'green' in newblobs:
				print "ONE OF EACH"
				if 'red' in newblobs:
					newRed = [round(x,0) for x in newblobs['red']]
					print "RED"

				if 'green' in newblobs:
					newGreen = [round(x,0) for x in newblobs['green']]
					print "GREEN"
				if 'blue' in newblobs:
					newBlue = [round(x,0) for x in newblobs['blue']]
					print "BLUE"
				if 'brown' in newblobs:
					newBrown = [round(x,0) for x in newblobs['brown']]
					print "PURPLE"
				if 'yellow' in newblobs:
					newYellow = [round(x,0) for x in newblobs['yellow']]
					print "YELLOW"	


				#RED -> X
				deltaRBX = newRed[0] - newBlue[0]
				deltaRBY= newRed[1] - newBlue[1]

				deltaRYX = newRed[0] - newYellow[0]
				deltaRYY= newRed[1] - newYellow[1]

				deltaRNX = newRed[0] - newBrown[0]
				deltaRNY= newRed[1] - newBrown[1]

				deltaRGX = newRed[0] - newGreen[0]
				deltaRGY= newRed[1] - newGreen[1]
				#BLUE -> X

				deltaBYX = newBlue[0] - newYellow[0]
				deltaBYY= newBlue[1] - newYellow[1]
				
				deltaBNX = newBlue[0] - newBrown[0]
				deltaBNY= newBlue[1] - newBrown[1]

				deltaBGX = newBlue[0] - newGreen[0]
				deltaBGY= newBlue[1] - newGreen[1]
				#YELLOW -> X

				deltaYNX = newYellow[0] - newBrown[0]
				deltaYNY= newYellow[1] - newBrown[1]				

				deltaYGX = newYellow[0] - newGreen[0]
				deltaYGY= newYellow[1] - newGreen[1]	

				#BROWN -> X
				deltaNGX = newBrown[0] - newGreen[0]
				deltaNGY= newBrown[1] - newGreen[1]	



				line = str(deltaRBX) + ","+str(deltaRBY)+","+\
					   str(deltaRYX) + ","+str(deltaRYY)+","+\
					   str(deltaRNX) + ","+str(deltaRNY)+","+\
					   str(deltaRGX) + ","+str(deltaRGY)+","+\
					   str(deltaBYX) + ","+str(deltaBYY)+","+\
					   str(deltaBNX) + ","+str(deltaBNY)+","+\
					   str(deltaBGX) + ","+str(deltaBGY)+","+\
					   str(deltaYNX) + ","+str(deltaYNY)+","+\
					   str(deltaYGX) + ","+str(deltaYGY)+","+\
					   str(deltaNGX) + ","+str(deltaNGY)+","+\
					   '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0'+"\n"
				# f.write(line)
				a = [deltaRBX,deltaRBY,deltaRYX,deltaRYY,deltaRNX,deltaRNY,deltaRGX,deltaRGY,
					deltaBYX,deltaBYY,deltaBNX,deltaBNY,deltaBGX,deltaBYY,
					deltaYNX,deltaYNY,deltaYGX,deltaYNY,
					deltaNGX,deltaNGY]
				r = net.activate(a)

				print "R IS: " + str(r) + "\n\n\n\n\n\n"
				
				maxx = -10
				ind = 0
				maxIndex2 = -2
				maxIndex = -1
				for x in r:
					if x > maxx:
						maxx = x
						maxIndex2 = maxIndex
						maxIndex = ind
					ind += 1
				simplecvimg.drawText(str([chr(maxIndex + 65), chr(maxIndex2 + 65)]),fontsize=25)
				# # # d = {0:'A',1:'B',2:'C',3:'D',4:'K',5:'L'}
				# # # simplecvimg.drawText(d[maxIndex],fontsize=25)
				# # 
				# if r[0] > r[1]:
				# 	simplecvimg.drawText("UP",fontsize=25)
				# else:
				# 	simplecvimg.drawText("DOWN",fontsize=25)
			# greenNail.save("a"+str(As)+'.png')
			
			f.close()

			(simplecvimg).show()

			c = cv.WaitKey(100)


if __name__ == '__main__':
	main()


	# 37 87 98