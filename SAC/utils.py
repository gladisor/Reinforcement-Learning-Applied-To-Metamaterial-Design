import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torchvision import transforms
from PIL import Image
import io

transform = transforms.Compose([
	transforms.Resize((50, 50)),
	transforms.Grayscale(),
	transforms.ToTensor()])

def getIMG(config, nCyl):
	"""
	Produces tensor image of configuration
	"""
	## Generate figure
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.axis('equal')
	ax.set_xlim(xmin=-6, xmax=6)
	ax.set_ylim(ymin=-6, ymax=6)
	ax.grid()
	
	coords = config.reshape(nCyl, 2)
	for cyl in range(nCyl):
		ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=1))

	## Convert to tensor
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	buf.seek(0)
	im = Image.open(buf)

	## Apply series of transformations
	X = transform(im)

	buf.close()
	plt.close(fig)
	return X.unsqueeze(0)