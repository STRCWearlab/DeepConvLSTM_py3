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


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, num_batches=-1):

	batch = lambda j : [x for x in range(j*batchsize,(j+1)*batchsize)]
	
	batches = [i for i in range(int(len(inputs)/batchsize)-1)]


	if shuffle:
		np.random.shuffle(batches)
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
		X, y = load_dataset('data/{}_data_{}'.format(name,i))
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


def iterate_minibatches_2D(inputs, targets, batchsize, seq_len, stride, shuffle=True, num_batches=-1, oversample=False,batchlen=10,val=False):

	assert (seq_len/stride).is_integer(), 'in order to generate sequential batches, the sliding window length must be divisible by the step.'

	starts = [[x for x in range(0,len(i)-int(((batchlen*seq_len)+1)/stride))] for i in inputs]

	for i in range(1,len(starts)):
		starts[i] = [x+1+starts[i-1][-1]+int(((batchlen*seq_len)+1)/stride) for x in starts[i]]


	starts = [val for sublist in starts for val in sublist]
	inputs = [val for sublist in inputs for val in sublist]
	targets = [val for sublist in targets for val in sublist]

	

	if batchlen > 1:

		step = lambda x : [int(x+i*seq_len/stride) for i in range(batchlen)]

		if shuffle:
			np.random.shuffle(starts)

		batches = np.empty((batchsize,batchlen),dtype=np.int32)

		if num_batches != -1:
			num_batches = int(num_batches*batchsize)


		for i,start in enumerate(starts[0:num_batches]):

			batch = np.array([i for i in step(start)],dtype=np.int32)


			if oversample and not any([targets[i] for i in batch]) and np.random.randint(10) < 8:
				pass
			else:
				batches[i%batchsize] = batch
				
				if i%batchsize == batchsize-1:

					batches = batches.transpose()
					
					for pos,batch in enumerate(batches):
						yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos
						batches = np.empty((batchsize,batchlen),dtype=np.int32)

		if val == True and num_batches == -1:
			for batch in batches:
				yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos


	elif batchlen == 1:

		if shuffle:
			np.random.shuffle(starts)

		batch = lambda j : [x for x in range(j,j+batchsize)]

		for j in starts[0:num_batches]:
			yield np.array([inputs[i] for i in batch(j)]), np.array([targets[i] for i in batch(j)])
