from datetime import datetime
from datetime import timedelta

def generateSRT(duration):
	print('Generating SRT...')
	curr_time = datetime.now()
	end_time = curr_time+timedelta(seconds=duration)
	_,ct = str(curr_time).split()
	_,et = str(end_time).split()
	ct,_ = ct.split('.')
	et,_ = et.split('.')
	file = open('srt.txt','a')
	file.write("INSERT VIDEO FILE'S NAME HERE \n")
	file.write(ct+' --> '+et+'\n')
	file.close()
	print('SRT generated')
	
