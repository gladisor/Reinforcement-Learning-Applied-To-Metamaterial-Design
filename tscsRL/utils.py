import json
import numpy as np

def dictToJson(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def jsonToDict(path):
	with open(path) as f:
		data = json.load(f)
	return data

def rtpairs(r, n):
	circle = []
	for i in range(len(r)):
		for j in range(n[i]):
			t = j*(2*np.pi/n[i])
			x = r[i] * np.cos(t)
			y = r[i] * np.sin(t)

			circle.append([x, y])
	return circle