import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from memory_profiler import profile
import time
import itertools
import _pickle as cp
from sliding_window import sliding_window
import os

def init_weights(m):
	if type(m) == nn.LSTM:
		for name, param in m.named_parameters():
			if 'weight_ih' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'weight_hh' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'bias' in name:
				param.data.fill_(0)
	elif type(m) == nn.Conv1d or type(m) == nn.Linear:
		torch.nn.init.orthogonal_(m.weight)
		m.bias.data.fill_(0)



def iterate_minibatches_batched(inputs, targets, batchsize, seq_len, stride, shuffle=True, num_batches=-1, oversample=False,batchlen=10,val=False):

	step = lambda x : [int(x+i*seq_len/stride) for i in range(batchlen)]

	if shuffle and batchlen > 0:
		# starts = [np.array([[x for x in range(j*(seq_len*batchlen+seq_len),(j*seq_len*batchlen)+(j+1)*seq_len)] for j in range(0,int((len(inputs)-seq_len)/(seq_len*batchlen+seq_len)))]) for inputs in inputs]
		starts = [[x for x in range(0,len(i)-int((batchlen*seq_len)+1/stride))] for i in inputs]

		for i in range(1,len(starts)):
			starts[i] = [x+1+starts[i-1][-1]+int((batchlen*seq_len)+1/stride) for x in starts[i]]


		starts = [val for sublist in starts for val in sublist]


		inputs = [val for sublist in inputs for val in sublist]
		targets = [val for sublist in targets for val in sublist]

		# np.random.shuffle(starts)


		xs = []
		ys = []

		if num_batches != -1:
			num_batches = num_batches*batchsize

		for start in starts[0:num_batches]:
			x = [inputs[i] for i in step(start)]
			y = [targets[i] for i in step(start)]

			if oversample and (all(y) == 0) and (np.random.randint(10) < 5):
				pass
			else:
				xs.append(x)
				ys.append(y)
				
				if len(xs) == batchsize:
					
					yield np.asarray(xs).transpose(1,0,2,3).squeeze(),np.asarray(ys).transpose().squeeze()
					xs = []
					ys = []

		if val == True:
			yield np.asarray(xs).transpose(1,0,2,3).squeeze(),np.asarray(ys).transpose().squeeze()
			xs = []
			ys = []

	# elif shuffle:
	# 	inputs = [val for sublist in inputs for val in sublist]
	# 	targets = [val for sublist in targets for val in sublist]

	# 	index = [i for i in range(len(inputs))]

	# 	np.random.shuffle(index)

	# 	np.array(index).reshape(batchsize,-1)

	# 	for batch in index:
	# 		yield [inputs[i] for i in batch], [targets[i] for i in batch]

	else:
		for start in range(int(len(inputs) - seq_len*batchsize + 1)):
			yield inputs[step(start)], targets[step(start)]

def iterate_minibatches(inputs, targets, batchsize, shuffle=True, num_batches=-1):

	batch = lambda j : [x for x in range(j*batchsize,(j+1)*batchsize)]
	
	batches = [i for i in range(int(len(inputs)/batchsize)-1)]


	if shuffle:
		# np.random.shuffle(batches)
		for i in batches[0:num_batches]:
			yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])

	else:
		for i in batches[0:num_batches]:
			yield np.array([inputs[i] for i in batch(i)]),np.array( [targets[i] for i in batch(i)])


def plot_data(logname='log.csv',save_fig='LASGFNO'):
	train_loss_plot = []
	val_loss_plot = []
	acc_plot = []
	f1_plot = []
	f1_macro = []

	with open(logname, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar= '|')
		for row in reader:
			train_loss_plot.append(float(row[0]))
			val_loss_plot.append(float(row[1]))
			acc_plot.append(float(row[2]))
			f1_plot.append(float(row[3]))
			f1_macro.append(float(row[4]))

	if save_fig:
		try:
			os.makedirs('Results/{}'.format(save_fig))
		except FileExistsError:
			pass


	plt.figure(1)
	plt.title('Training loss')
	plt.ylabel('Categorical cross entropy')
	plt.xlabel('Epoch')
	plt.plot(train_loss_plot)
	if save_fig:
		plt.savefig('Results/{}/Train_loss_{}.png'.format(save_fig,time.time()))

	plt.figure(2)
	plt.title('Validation loss')
	plt.ylabel('Categorical cross entropy')
	plt.xlabel('Epoch')
	plt.plot(val_loss_plot)
	if save_fig:
		plt.savefig('Results/{}/Val_loss_{}.png'.format(save_fig,time.time()))

	plt.figure(3)
	plt.title('Validation Accuracy')
	plt.ylabel('Weighted accuracy')
	plt.xlabel('Epoch')
	plt.plot(acc_plot)
	if save_fig:
		plt.savefig('Results/{}/Val_acc_{}.png'.format(save_fig,time.time()))

	plt.figure(4)
	plt.title('Validation f1 score')
	plt.ylabel('f1 score')
	plt.xlabel('Epoch')
	plt.plot(f1_plot,label='Weighted')
	plt.plot(f1_macro,label='Macro')
	plt.legend()
	if save_fig:
		plt.savefig('Results/{}/Val_f1_{}.png'.format(save_fig,time.time()))

	if not save_fig:
		plt.show()

	
def load_opp_runs(name,num_files,len_seq, stride):
	Xs = []
	ys = []

	for i in range(num_files):
		X, y = load_dataset('/data/{}_data_{}'.format(name,i))
		X, y = opp_slide(X, y, len_seq, stride, save=False)
		Xs.append(X)
		ys.append(y)

	return Xs, ys


def load_dataset(filename):

	with open(filename, 'rb') as f:
		data = cp.load(f)

	X, y = data

	print('Got {} samples from {}'.format(X.shape, filename))

	X = X.astype(np.float32)
	y = y.astype(np.uint8)


	return X, y

def opp_slide(data_x, data_y, ws, ss,save=False):
	x = sliding_window(data_x, (ws,data_x.shape[1]),(ss,1))
	y = np.asarray([i[-1] for i in sliding_window(data_y, ws, ss)]).astype(np.uint8)

	if save:
		with open('data_slid','wb') as f:
			cp.dump((x,y),f,protocol=4)

	else:
		return x,y


plot_data('log.csv')
