import json
import numpy as np
import matplotlib.pyplot as plt


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
            t = j * (2 * np.pi / n[i])
            x = r[i] * np.cos(t)
            y = r[i] * np.sin(t)

            circle.append([x, y])
    return circle


def plot(filename, data, path):
    plt.figure()
    plt.plot(data, 'r')
    plt.title(filename)
    plt.xlabel('Epochs')
    plt.ylabel(filename)
    plt.savefig(path + filename)
