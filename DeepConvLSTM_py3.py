import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,confusion_matrix
from datetime import datetime
from torch import nn
from utils import *
import csv
import seaborn as sn
import argparse

train_on_gpu = torch.cuda.is_available() # Check for cuda

# Define constants


n_channels = 113 # number of sensor channels
window_size = 30 # Sliding window length
stride = 1 # Sliding window step
num_epochs = 300 # Max no. of epochs to train for
num_batches= 20 # No. of training batches per epoch. -1 means all windows will be presented at least once, up to batchlen times per epoch (unless undersampled)
batch_size = 1000 # Batch size / width - this many windows of data will be processed at once
patience= 200 # Patience of early stopping routine. If criteria does not decrease in this many epochs, training is stopped.
batchlen = 50 # No. of consecutive windows in a batch. If false, the largest number of windows possible is used.
val_batch_size = 1000 # Batch size for validation/testing. 
test_batch_size = 1000 # Useful to make this as large as possible given GPU memory, to speed up testing.
lr = 0.0001 # Initial (max) learning rate
num_batches_val = 1 # How many batches should we validate on each epoch
lr_step = 100
n_conv = 4
n_filters = 64
log_name = 'log'

opp_class_names = ['Null','Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge',
'Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2',
'Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']


## Define our DeepConvLSTM class, subclassing nn.Module.
class DeepConvLSTM(nn.Module):

	def __init__(self, n_conv = 4, n_hidden = 128, n_layers = 2, n_filters = 64,
				n_classes = 18, filter_size = 5,pool_filter_size=3, drop_prob = 0.5, 
				window_size = 30, n_channels = 113):

		super(DeepConvLSTM, self).__init__() # Call init function for nn.Module whenever this function is called

		self.__dict__.update(locals())

		# Convolutional net
		self.convlayer = nn.ModuleList([nn.Conv1d(n_channels, n_filters, (filter_size))]) #First layer should map from number of channels to number of filters
		self.convlayer.extend([nn.Conv1d(n_filters, n_filters, (filter_size)) for i in range(n_conv-1)]) # Subsequent layers should map n_filters -> n_filters

		# LSTM layers
		if self.n_layers > 0:
			self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
			self.predictor = nn.Linear(n_hidden,n_classes)
		else:
			self.predictor = nn.Linear(n_filters,n_classes)
		
		# Dropout layer
		self.dropout = nn.Dropout(p=drop_prob)


	
	def forward(self, x, hidden, batch_size):

		#Reshape x if necessary to add the 2nd dimension
		x = x.view(-1, self.n_channels, self.window_size)
		# print(x.size())

		for conv in self.convlayer:
			x = conv(x)

		x = x.view(batch_size, -1, self.n_filters)
		
		if model.n_layers > 0:
			x,hidden = self.lstm(x, hidden)

			x = self.dropout(x)

			x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
		

		out = self.predictor(x)

		return out, hidden

	def init_hidden(self, batch_size):

		weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
		
		if (train_on_gpu):
			## Generate new tensors of zeros with similar type to weight, but different size.
			hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(), # Hidden state
				  weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda()) # Cell state
		else:
			hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
					  weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
		
		return hidden



def train(net, X_train, y_train,X_val,y_val, epochs=num_epochs,
	lr=lr, time=True, shuffle=True, log_name=log_name+'.csv',
	batch_size=batch_size, batchlen=batchlen, num_batches=num_batches, 
	window_size=window_size, val_batch_size=val_batch_size, 
	num_batches_val=num_batches_val, patience=patience):

	if time:
		print('Starting training at',datetime.now())
		start_time=datetime.now()

	# num_batches= int(500000/(batch_size * batchlen))
	# print(num_batches)

	opt = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=3e-8,amsgrad=True)
	scheduler = torch.optim.lr_scheduler.StepLR(opt,lr_step) # Learning rate scheduler to reduce LR every 100 epochs

	if(train_on_gpu):
		net.cuda()

	train_stats = np.unique([a for y in y_train for a in y],return_counts=True)[1]
	val_stats = np.unique([a for y in y_val for a in y],return_counts=True)[1]

	print('Training set statistics:')
	print(len(train_stats),'classes with distribution',train_stats)
	print('Validation set statistics:')
	print(len(val_stats),'classes with distribution',val_stats)

	weights = torch.tensor([max(train_stats)/i for i in train_stats],dtype=torch.float)

	if train_on_gpu:
		weights = weights.cuda()
	
	criterion = nn.CrossEntropyLoss(weight=weights) # Prepare weighted cross entropy for training and validation.
	val_criterion = nn.CrossEntropyLoss()

	early_stopping = EarlyStopping(patience=patience, verbose=False)

	print('Logging training to',log_name)
	with open(log_name, 'w', newline='') as csvfile: # We will save some training statistics to plot a loss curve later.

		for e in range(epochs):
			
			train_losses = []
			net.train() # Setup network for training



			for batch in iterate_minibatches_2D(X_train, y_train, batch_size, stride, num_batches=num_batches, batchlen=batchlen, shuffle=shuffle, drop_last=True):

				x,y,pos= batch


				inputs, targets = torch.from_numpy(x), torch.from_numpy(y)				

				opt.zero_grad()  # Clear gradients in optimizer


				if pos==0:
					h = net.init_hidden(inputs.size()[0])

				h = tuple([each.data for each in h])  # Get rid of gradients in hidden and cell states

		
				if train_on_gpu:
					inputs,targets = inputs.cuda(),targets.cuda()
				
				output, h = net(inputs,h,inputs.size()[0]) # Run inputs through network


				loss = criterion(output, targets.long())
				
				loss.backward()
				opt.step()	

				train_losses.append(loss.item())

		
			val_losses = []
			net.eval() # Setup network for evaluation

			top_classes = []
			targets_cumulative = []

			with torch.no_grad():
				for batch in iterate_minibatches_2D(X_val, y_val, val_batch_size, stride, shuffle=shuffle, batchlen=batchlen, drop_last=False):
					

					x,y,pos=batch

					inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
					
					targets_cumulative.extend([y for y in y])

					if pos == 0:
						val_h = net.init_hidden(inputs.size()[0])

					if train_on_gpu:
						inputs,targets = inputs.cuda(),targets.cuda()
					
					output, val_h = net(inputs,val_h,inputs.size()[0])
	
					val_loss = val_criterion(output, targets.long())
					val_losses.append(val_loss.item())


					top_p, top_class = output.topk(1,dim=1)
					top_classes.extend([top_class.item() for top_class in top_class.cpu()])

			equals = [top_classes[i] == target for i,target in enumerate(targets_cumulative)]
			accuracy = np.mean(equals)

			f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
			f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
			stopping_metric = (f1score+f1macro+accuracy) - np.mean(val_losses)
			
			scheduler.step()

			print('Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.2f}, f1: {:.2f}, M f1: {:.2f}, M: {:.4f}'.format(e+1,epochs,np.mean(train_losses),np.mean(val_losses),accuracy,f1score,f1macro,stopping_metric))
		

			early_stopping((-stopping_metric),net)

			writer = csv.writer(csvfile, delimiter=' ',
									quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow([np.mean(train_losses),np.mean(val_losses),accuracy,f1score,f1macro])
			
			

			if early_stopping.early_stop:
				print("Stopping training, validation metric has not decreased in {} epochs.".format(patience))
				break



			
			
	print('Training finished at ',datetime.now())
	print('Total time elapsed during training:',(datetime.now()-start_time).total_seconds())



def test(net, X_test, y_test, batch_size, shuffle=True, run_name='log', show_results=False, **kwargs):

	print('Starting testing at', datetime.now())
	start_time=datetime.now()
	criterion = nn.CrossEntropyLoss()
	
	if(torch.cuda.is_available()):
		net.cuda()

	net.eval()

	val_losses = []
	accuracy=0
	f1score=0
	f1macro=0
	targets_cumulative = []
	top_classes = []



	with torch.no_grad():
			
		for batch in iterate_minibatches_test(X_test, y_test, window_size, stride):
				
			x,y,pos=batch

			inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

			targets_cumulative.extend([y for y in y])

			if(train_on_gpu):
				targets,inputs = targets.cuda(),inputs.cuda()

			if pos == 0:
				test_h = net.init_hidden(inputs.size()[0])

			output, test_h = net(inputs,test_h,inputs.size()[0])

			val_loss = criterion(output, targets.long())
			val_losses.append(val_loss.item())

			top_p, top_class = output.topk(1,dim=1)
			top_classes.extend([p.item() for p in top_class])

	equals = [top_classes[i] == target for i,target in enumerate(targets_cumulative)]
	accuracy = np.mean(equals)

	print('Finished testing at', datetime.now())
	print('Total time elapsed during inference:', (datetime.now()-start_time).total_seconds())

	f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')

	classreport = classification_report(targets_cumulative, top_classes,target_names=opp_class_names)
	confmatrix = confusion_matrix(targets_cumulative, top_classes,normalize='true')
	print('---- TESTING REPORT {} ----'.format(run_name))
	print(classreport)

	df_cm = pd.DataFrame(confmatrix, index=opp_class_names,columns=opp_class_names)
	plt.figure(10,figsize=(15,12))
	sn.heatmap(df_cm,annot=True,fmt='.2f',cmap='Purples')
	plt.savefig('Results/{}-{}-{}-Confmatrix.png'.format(datetime.now(),log_name,f1score))
	if show_results:
		plt.show(plt.figure(10))

	print('Testing accuracy:',accuracy)
	print('Testing f1 score:',f1score)




###Command line parser - change constants if necessary
def get_args():
	'''This function parses and return arguments passed in'''
	parser = argparse.ArgumentParser(
		description='DeepConvLSTM')
	# Add arguments
	parser.add_argument(
		'-a','--architecture', type=str, help='Desired network architecture. Defaults to original DeepConvLSTM.', required=False, default='HASCA-standard')
	parser.add_argument(
		'-l','--log_name', type=str, help='Name of file to write training log to. Also determines name of graphs and confusion matrix. Defaults to log.', required=False, default='log')
	parser.add_argument(
		'-w','--window_size', type=int, help='Desired length of sliding window. Defaults to 30.', required=False, default=30)
	parser.add_argument(
		'-bl','--batch_length', type=int, help='Length of metabatches / number of consecutive windows. Defaults to 50.', required=False, default=50)
	parser.add_argument(
		'-bw','--batch_width', type=int, help='Width of metabatches / number of windows per parallel batch. Defaults to 1000.', required=False, default=1000)
	parser.add_argument(
		'-n','--num_batches', type=int, help='Number of batches per epoch. Defaults to 20', required=False, default=20)
	parser.add_argument(
		'-vn','--num_batches_val', type=int, help='Number of batches for validation per epoch. Defaults to 1', required=False, default=1)
	parser.add_argument(
		'-p','--early_stopping_patience', type=int, help='Number of epochs after which training is stopped if no improvement in validation metric is made. Defaults to 20.', required=False, default=20)
	
	# Array for all arguments passed to script
	args = parser.parse_args()
	# Assign args to variables
	architecture = args.architecture
	window_size = args.window_size
	log_name = args.log_name
	# Return all variable values

	## Architecture definitions
	HASCA_short = {'window_size':10,'n_conv':2,'n_filters':128}
	HASCA_standard = {'window_size':window_size,'n_conv':4,'n_filters':64}
	HASCA_challenge = {'window_size':window_size,'n_conv':4,'n_filters':64,'filter_size':7}

	## Select architecture from above.
	if architecture == 'HASCA-short':
		architecture = HASCA_short
	if architecture == 'HASCA-standard':
		architecture = HASCA_standard
	if architecture == 'HASCA-challenge':
		architecture = HASCA_challenge

	arguments = {'window_size':args.window_size,'batchlen':args.batch_length,'batch_size':args.batch_width,
	'num_batches':args.num_batches,'num_batches_val':args.num_batches_val,'log_name':log_name+'.csv',
	'patience':args.early_stopping_patience}


	return architecture, arguments



if __name__ == '__main__':

	arch, args = get_args()	

	print('==HYPERPARAMETERS==')
	print(args,arch)

	X_train,y_train = load_data('train',args['window_size'], stride)
	X_val, y_val = load_data('val',args['window_size'], stride)

	net = DeepConvLSTM(**arch)

	net.apply(init_weights)


	try:
		train(net,X_train,y_train,X_val,y_val,**args)
	except(KeyboardInterrupt):
		pass

	del X_train
	del y_train
	del X_val
	del y_val

	plot_data(logname=log_name,save_fig='{}-{}-{}-{}'.format(log_name,datetime.now(),args['window_size'],args['batchlen']))

	print('Loading test data')

	X_test, y_test = load_data('test',args['window_size'], stride)

	print('Testing fully trained model.')

	test(net, X_test,y_test, **args)

	torch.save(net.state_dict(), 'fullytrained.pt')

	print('Loading model state_dict from checkpoint.')

	state_dict = torch.load('checkpoint.pt')

	net.load_state_dict(state_dict)

	print('Testing checkpointed model.')

	test(net, X_test,y_test, batch_size=test_batch_size, **args)

	print('Plotting training curves')

