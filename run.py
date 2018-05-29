from srt import generateSRT

def run(action = '1', time = 0):
	while action != 'exit':
		action, time = input().split()
		if action == '1':
			print("Attentive")
		elif action == '0':
			time = int(time)
			if(time > 5):
				print("Distracted")
				generateSRT(time-5)


if __name__ == '__main__':
	run()
