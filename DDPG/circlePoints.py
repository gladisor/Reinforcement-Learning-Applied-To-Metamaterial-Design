import numpy as np
import matplotlib.pyplot as plt

def rtpairs(r, n):
	circle = []
	for i in range(len(r)):
		for j in range(n[i]):
			t = j*(2*np.pi/n[i])
			x = r[i] * np.cos(t)
			y = r[i] * np.sin(t)

			circle.append([x, y])
	return circle

if __name__ == '__main__':
	T = [1, 10, 20, 30, 40, 50, 60]
	R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]