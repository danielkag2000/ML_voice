from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from sys import argv

label_num = 30
window_size = 3
epochs = 100
seed = 19
module_file = "cnn_data"
debug = True


def create_conv2d(size_in, ch_in, ch_out, ker, stride=1, device=None):
	"""
	Creates a convolution instance with the given parameters and
	returns the new output size based on the parameters
	:param size_in: the input size
	:param ch_in: the number of input channels
	:param ch_out: the number of output channels
	:param ker: the kernel size
	:param stride: the stride
	:param device: the device
	:return:
			conv:       the matrix of the conv
			size_new:   the new size of the output
	"""
	# make the conv
	conv = nn.Conv2d(ch_in, ch_out, kernel_size=ker, stride=stride)
	# to run on a device
	if device:
		conv = conv.to(device)

	# calculate the new size
	size_new = [(size - ker) // stride + 1 for size in size_in]

	return conv, size_new


class CNN(nn.Module):
	def __init__(self, vid_size, device):
		"""
		:param vid_size: the input size
		:param device: the device to run on
		"""
		super(CNN, self).__init__()

		# create conv1
		self.conv1, size = create_conv2d(vid_size, 1, 6, 10, device=device)

		# for the pooling
		size = [s // window_size for s in size]
		# create conv2
		self.conv2, size = create_conv2d(size, 6, 16, 5, device=device)

		# for the pooling
		size = [s // window_size for s in size]

		# define the layers (to be linear)
		# first (the size after conv2 and second pooling X 120)
		self.fc0 = nn.Linear(size[0] * size[1] * 16, 120).to(device)

		# second (120 X 84)
		self.fc1 = nn.Linear(120, 84).to(device)

		# third (84 X label_num=30)
		self.fc2 = nn.Linear(84, label_num).to(device)

	def forward(self, x):
		"""
		Do forward propagation
		:param x the example
		:return: the output of the last layer (with softmax)
		"""
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), window_size)
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), window_size)
		x = x.view(-1, self.get_vector_size(x))
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		# return F.log_softmax(x, dim=1)
		return x

	@staticmethod
	def get_vector_size(x):
		"""
		Get the vector size
		:param x the example
		:return: the size of the flat vector (the size of the 1D array of the vector)
		"""
		number_of_features = 1
		size = x.size()[1:]  # all dimensions except the batch dimension
		for s in size:
			number_of_features *= s
		return number_of_features


def train(n_epochs, model, train_loader, optimizer, device):
	"""
	Train the model
	:param n_epochs the number of epochs to train with the dataset
	:param model the model
	:param train_loader the train loader
	:param optimizer the optimizer
	:param device the device
	"""
	if debug:
		print("Started Training")
	# run epochs times
	for epoch in range(n_epochs):
		if debug:
			print("epoch:", epoch)
		# train the model
		model.train()
		for batch_idx, (data, labels) in enumerate(train_loader):
			data = data.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			output = model(data)
			# loss = F.nll_loss(output, labels)
			loss = F.cross_entropy(output, labels)
			loss.backward()
			optimizer.step()
	if debug:
		print("Finished Training")


def load_module(model):
	"""
	Loads a model from the file
	:param model: te model itself
	:return: nothing, void
	"""
	if os.path.isfile(module_file):
		if debug:
			print("Started Loading Module")
		model.load_state_dict(torch.load(module_file))
		if debug:
			print("Finished Loading Module")
	else:
		print("Couldn't Load Module, File '{}' Doesn't Exists".format(load_module))
		exit()


def validate(model, val_loader, device):
	"""
	Validate the model
	:param model the model
	:param val_loader the validation loader
	:param device the device
	"""
	if debug:
		print("Started Validating")

	correct = 0
	total = 0

	for i, (inputs, labels) in enumerate(val_loader):
		inputs, labels = inputs.to(device), labels.to(device)
		output = model(inputs)
		correct += output.argmax(dim=1).eq(labels).sum().item()
		total += len(output)

	if debug:
		print("Finished Validating")
	print('Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))

	return 100. * correct / total


def test(model, test_loader, output_file):
	"""
	Validate the model
	:param model: the model
	:param test_loader: the testing loader
	:param output_file: the output file to write all the classifications
	"""
	if debug:
		print("Started Testing")
	model.to("cpu")
	model.eval()
	# get the paths to all the testing files
	paths = test_loader.dataset.spects
	paths = [os.path.basename(p[0]) for p in paths]
	# get the files themselves
	inputs = [x[0] for x in test_loader]
	# if it's the first one to test
	first = True

	# open the test output file
	with open(output_file, "w") as f:
		# go over each input file
		for x, path in zip(inputs, paths):
			# do classification
			value, classification = torch.max(model.forward(x), 1)
			# write the classification into the output file
			if not first:
				f.write("\n")
			f.write(path + ", " + str(classification.item()))
			first = False

	if debug:
		print("Finished Testing")


def main():
	np.random.seed(seed)
	torch.manual_seed(seed)

	# get the data set
	dataset = GCommandLoader('./data/train')

	# create the train loader
	train_loader = torch.utils.data.DataLoader(
		dataset, batch_size=100, shuffle=True,
		num_workers=8, pin_memory=True, sampler=None)

	# get the validation data set
	valid_set = GCommandLoader('./data/valid')

	# create the validation loader
	valid_loader = torch.utils.data.DataLoader(
		valid_set, batch_size=100, shuffle=False,
		num_workers=8, pin_memory=True, sampler=None)

	# get the validation data set
	test_set = GCommandLoader('./data/test')

	# create the validation loader
	test_loader = torch.utils.data.DataLoader(
		test_set, batch_size=1, shuffle=False,
		num_workers=8, pin_memory=True, sampler=None)

	# use cuda (for running on the gpu)
	cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if cuda else "cpu")

	# create the model
	model = CNN([161, 101], device)
	# model = CNN(device)

	# create the optimizer
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	optimizer = optim.Adam(model.parameters())

	# -- TRAIN -- #
	if len(argv) <= 1 or argv[1] == "train":
		train(epochs, model, train_loader, optimizer, device)
		# -- SAVE -- #
		torch.save(model.state_dict(), module_file)
	else:
		load_module(model)

	# -- VALIDATION -- #
	validate(model, valid_loader, device)

	# -- TEST -- #
	if len(argv) <= 1 or argv[1] == "test":
		test(model, test_loader, "test_y")


if __name__ == "__main__":
	print("Started")
	main()
	print("Fnished")
