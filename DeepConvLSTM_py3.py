
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np 

import matplotlib.pyplot as plt
import pandas as pd 
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,confusion_matrix
from datetime import datetime
import time
import os
import math 
from torch import nn
import torch.nn.functional as F 
from utils import *
import csv
from pytorchtools import EarlyStopping


# Define constants
n_channels = 67 # number of sensor channels
len_seq = 32 # Sliding window length MUST BE A MULTIPLE OF 4 +1
stride = 1 # Sliding window step
num_epochs = 300 # Max no. of epochs to train for
num_batches= -1 # No. of batches per epoc. -1 means all windows will be presented at least once, up to batchlen times per epoch (unless undersampled)
batch_size = 1000 # Batch size / width - this many sequential batches will be processed at once
patience= 50 # Patience of early stopping routine. If criteria does not decrease in this many epochs, training is stopped.
batchlen = 10 # No. of consecutive
val_batch_size = 10000 # Batch size for validation/testing. Useful to make this as large as possible given GPU memory, to speed up validation and testing.
test_batch_size = 10000




opp_class_names = ['Null','Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge',
'Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2',
'Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']







class DeepConvLSTM(nn.Module):

	def __init__(self, n_hidden = 128, n_layers = 2, n_filters = 64,
				n_classes = 18, filter_size = 5,pool_filter_size=3, drop_prob = 0.5):

		super(DeepConvLSTM, self).__init__() # Call init function for nn.Module whenever this function is called

		self.drop_prob = drop_prob # Dropout probability
		self.n_layers = n_layers # Number of layers in the lstm network
		self.n_hidden = n_hidden # number of hidden units per layer in the lstm
		self.n_filters = n_filters # number of convolutional filters per layer
		self.n_classes = n_classes # number of target classes
		self.filter_size = filter_size # convolutional filter size
 
 		# Convolutional net
		self.convlayer = nn.Sequential(
			nn.Conv2d(n_channels, n_filters, (filter_size,1)),
			# nn.MaxPool2d((pool_filter_size,1)),
			nn.Conv2d(n_filters, n_filters, (filter_size,1)),
			# nn.MaxPool2d((pool_filter_size,1)),
			nn.Conv2d(n_filters, n_filters, (filter_size,1)),
			nn.Conv2d(n_filters, n_filters, (filter_size,1))
			)

		# LSTM layers
		self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)

		# Dropout layer
		self.dropout = nn.Dropout2d(p=drop_prob)
		self.dropout2 = nn.Dropout(p=drop_prob)

		# Output layer
		self.predictor = nn.Linear(n_hidden,n_classes)

	
	def forward(self, x, hidden, batch_size):

		#Reshape x if necessary to add the 2nd dimension
		x = x.view(-1, n_channels, len_seq, 1)

		x = self.convlayer(x)
		x = self.dropout(x)
		x = x.view(batch_size, -1, self.n_filters)
		

		x,hidden = self.lstm(x, hidden)
		x = self.dropout2(x)

		x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
		out = self.predictor(x)

		return out, hidden

	def init_hidden(self, batch_size):

		weight = next(self.parameters()).data
		
		if (train_on_gpu):
			hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
				  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
		else:
			hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
					  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
		
		return hidden



def train(net, X_train, y_train,X_val,y_val, epochs=num_epochs, batch_size=batch_size, lr=0.01, remove_nulls=False, time=True, shuffle=True):

	if time:
		print('Starting training at',datetime.now())
		start_time=datetime.now()

	opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True) #Stochastic gradient descent optimiser with nesterov momentum
	scheduler = torch.optim.lr_scheduler.StepLR(opt,100) # Learning rate scheduler to reduce LR every 100 epochs

	criterion = nn.CrossEntropyLoss()

	if(train_on_gpu):
		net.cuda()

	print('Validation set statistics:')
	print(len(np.unique([a for y in y_val for a in y])),np.unique([a for y in y_val for a in y],return_counts=True)[1])
	print('Training set statistics:')
	print(len(np.unique([a for y in y_train for a in y])),np.unique([a for y in y_train for a in y],return_counts=True)[1])

	early_stopping = EarlyStopping(patience=patience, verbose=False)

	with open('log.csv', 'w', newline='') as csvfile:

		for e in range(epochs):
			
			train_losses = []
			net.train()

			h = net.init_hidden(batch_size)

			for batch in iterate_minibatches_batched(X_train, y_train, batch_size, len_seq, stride, shuffle=True, num_batches=num_batches, oversample=True, batchlen=batchlen):
				

				x,y= batch

				inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
				
				
				if(train_on_gpu):
					inputs, targets = inputs.cuda(), targets.cuda()
				
				

				opt.zero_grad() # Clear gradients in opt

				h = tuple([each.data for each in h])  # Get rid of gradients in hidden var from previous states

				output = torch.FloatTensor().cuda()
				targets_cumulative = torch.ByteTensor().cuda()

				for j,i in enumerate(inputs):

					
					i = i.reshape((-1, len_seq, n_channels, 1))
					out, h = net(i,h,inputs.size()[1])
					output = torch.cat((output,out))
					targets_cumulative = torch.cat((targets_cumulative, targets[j]))

					

				loss = criterion(output, targets_cumulative.long())
				train_losses.append(loss.item())
				loss.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)

				opt.step()	


		
			val_losses = []
			val_losses_weighted = []
			net.eval()

			top_classes = []
			targets_cumulative = []
			



			with torch.no_grad():
				for batch in iterate_minibatches_batched(X_val, y_val, val_batch_size, len_seq, stride, shuffle=True, num_batches=-1, batchlen=batchlen, val=True):
					

					x,y=batch
						

					inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
					
					targets_cumulative.extend([y for y in y for y in y])
					

					if(train_on_gpu):
						inputs, targets = inputs.cuda(), targets.cuda()

					val_h = net.init_hidden(inputs.size()[1])
					val_h = tuple([each.data for each in val_h])


					for j, i in enumerate(inputs):
						
						x = x.reshape((-1, len_seq, n_channels, 1))
						output, val_h = net(i,val_h,inputs.size()[1])
		
						val_loss = criterion(output, targets[j].long())
						val_losses.append(val_loss.item())

						top_p, top_class = output.topk(1,dim=1)
						top_classes.extend([top_class.item() for top_class in top_class.cpu()])

			equals = [top_classes[i] == target for i,target in enumerate(targets_cumulative)]
			accuracy = np.mean(equals)

			f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
			f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
			balacc = metrics.balanced_accuracy_score(targets_cumulative, top_classes, adjusted=True)

			
			scheduler.step()

			stopping_metric = (f1score+balacc-np.mean(val_losses))

			print('Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Val f1: {:.4f}, M f1: {:.4f}, bal: {:.4f}, Metric: {:.4f}'.format(e+1,epochs,np.mean(train_losses),np.mean(val_losses),f1score,f1macro,balacc,stopping_metric))
		

			early_stopping((-stopping_metric), net)

			writer = csv.writer(csvfile, delimiter=' ',
									quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow([np.mean(train_losses),np.mean(val_losses),accuracy,f1score,f1macro])
			
			

			if early_stopping.early_stop:
				print("Stopping training, validation loss has not decreased in {} epochs.".format(patience))
				break



			
			
	print('Training finished at ',datetime.now())
	print('Total time elapsed during training:',(datetime.now()-start_time).total_seconds())



def test(net, X_test, y_test, batch_size, remove_nulls=False, shuffle=True):

	print('Starting inference at', datetime.now())
	start_time=datetime.now()
	criterion = nn.CrossEntropyLoss()
	
	if(train_on_gpu):
		net.cuda()

	net.eval()

	val_losses = []
	accuracy=0
	f1score=0
	f1macro=0
	targets_cumulative = []
	top_classes = []

	with torch.no_grad():

		for batch in iterate_minibatches_batched(X_test, y_test, test_batch_size, len_seq, stride, shuffle=True, num_batches=-1, batchlen=batchlen, val=True):

			
				
			x,y=batch


			
			inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

			targets_cumulative.extend([y for y in y for y in y])


			

			if(train_on_gpu):
				inputs, targets = inputs.cuda(), targets.cuda()


			test_h = net.init_hidden(inputs.size()[1])
			test_h = tuple([each.data for each in test_h])

			for j, i in enumerate(inputs):

				x = x.reshape((-1, len_seq, n_channels, 1))

				output, test_h = net(i,test_h,inputs.size()[1])

				val_loss = criterion(output, targets[j].long())
				val_losses.append(val_loss.item())

				top_p, top_class = output.topk(1,dim=1)
				top_classes.extend([p.item() for p in top_class])




	print('Finished inference at', datetime.now())
	print('Total time elapsed during inference:', (datetime.now()-start_time).total_seconds())

	f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
	balacc = metrics.balanced_accuracy_score(targets_cumulative, top_classes,adjusted=True)

	classreport = classification_report(targets_cumulative, top_classes,target_names=opp_class_names)
	confmatrix = confusion_matrix(targets_cumulative, top_classes)
	print('---- TESTING REPORT ----')
	print(classreport)
	print('---- CONFUSION MATRIX ----')
	print(confmatrix)

	print('Testing balanced acc:',balacc)


if __name__ == '__main__':


	X_train,y_train = load_opp_runs('train',12,len_seq, stride)
	X_val, y_val = load_opp_runs('val',3,len_seq, stride)

	net = DeepConvLSTM()

	net.apply(init_weights)

	train_on_gpu = torch.cuda.is_available() # Check for cuda

	try:
		train(net,X_train,y_train,X_val,y_val, remove_nulls=False)
	except(KeyboardInterrupt):
		pass

	del X_train
	del y_train
	del X_val
	del y_val

	print('Loading test data')

	X_test, y_test = load_opp_runs('test',5,len_seq, stride)

	print('Testing and saving fully trained model.')

	test(net, X_test,y_test, batch_size=batch_size, remove_nulls=False)

	torch.save(net.state_dict(), 'fullytrained.pt')

	print('Loading model state_dict from checkpoint.')

	state_dict = torch.load('checkpoint.pt')

	net.load_state_dict(state_dict)

	print('Testing checkpointed model.')

	test(net, X_test,y_test, batch_size=batch_size, remove_nulls=False)

	print('Plotting training curves')

	plot_data(save_fig='{}-{}-{}'.format(len_seq,stride,num_epochs))
