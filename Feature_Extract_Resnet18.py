import torch
from torchvision import transforms
import config

class Resnet18Feature:
	def __init__(self, PATH):
		# store the number of bins the histogram will use
		self.model = torch.load(PATH)

	def describe(self, image):
		transformer = transforms.Compose([
			transforms.Resize(config.input_size),
			transforms.CenterCrop(config.input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		device = torch.device("cpu")
		transformer(image)
		image = torch.tensor(image)
		image = image.to(device)
		feature = self.model(image)

		return feature.cpu().data.numpy().flatten()
