import numpy as np
from numpy import genfromtxt as gft 
import matplotlib.pyplot as plt 
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC


file = open("classifier.pkl","rb")
clf = pickle.load(file)
file.close()

data = gft("./Dataset.csv", delimiter = ',')
mydata = data[2:,1:9]


time = []
value1 = []
value2 = []
value3 = []
value4 = []
value5 = []
value6 = []
value7 = []
value8 = []

fig = plt.figure()

for f in range(len(data)):
	time.append(f)
	value1.append(mydata[f,0])
	value2.append(mydata[f,1])
	value3.append(mydata[f,2])
	value4.append(mydata[f,3])
	value5.append(mydata[f,4])
	value6.append(mydata[f,5])
	value7.append(mydata[f,6])
	value8.append(mydata[f,7])
	sample = mydata[f].reshape(1,-1)
	predicted = clf.predict(sample)

	displayText = "Attentive"

	if(predicted[0] == 0):
		displayText = "Distracted"

	

	

	if f % 2000 == 0:
		print(chr(27)+"[2J")
		plt.subplot(3,3,1)
		plt.plot(time,value1, color = 'red')

		plt.subplot(3,3,2)
		plt.plot(time,value2, color = 'green')

		plt.subplot(3,3,3)
		plt.plot(time,value3, color = 'yellow')

		plt.subplot(3,3,4)
		plt.plot(time,value4, color = 'blue')

		plt.subplot(3,3,5)
		plt.plot(time,value5, color = 'cyan')

		plt.subplot(3,3,6)
		plt.plot(time,value6, color = 'maroon')

		plt.subplot(3,3,7)
		plt.plot(time,value7, color = 'black')

		plt.subplot(3,3,8)
		plt.plot(time,value8, color = 'orange')

		print(displayText)
		

		plt.draw()
		plt.pause(0.0001)


