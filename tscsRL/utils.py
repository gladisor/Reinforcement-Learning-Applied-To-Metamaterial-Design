import json

def dictToJson(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def jsonToDict(path):
	with open(path) as f:
		data = json.load(f)
	return data