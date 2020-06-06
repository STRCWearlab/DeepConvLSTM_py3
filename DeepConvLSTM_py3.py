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

# Define constants

n_channels = 113 # number of sensor channels
len_seq = 24 # Sliding window length
stride = 8 # Sliding window step
num_epochs = 300 # Max no. of epochs to train for
num_batches= 5 # No. of training batches per epoch. -1 means all windows will be presented at least once, up to batchlen times per epoch (unless undersampled)
batch_size = 1000 # Batch size / width - this many windows of data will be processed at once
patience= 300 # Patience of early stopping routine. If criteria does not decrease in this many epochs, training is stopped.
batchlen = 500 # No. of consecutive windows in a batch. If false, the largest number of windows possible is used.
val_batch_size = 1000 # Batch size for validation/testing. 
test_batch_size = 10000 # Useful to make this as large as possible given GPU memory, to speed up testing.
lr = 0.0001 # Initial (max) learning rate
num_batches_val = 1 # How many batches should we validate on each epoch
lr_step = 100

opp_class_names = ['Null','Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge',
'Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2',
'Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']



## Define our DeepConvLSTM class, subclassing nn.Module.
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
		self.pool_filter_size = pool_filter_size # max pool filter size if using
 
 		# Convolutional net
		self.convlayer = nn.Sequential(
			nn.Conv1d(n_channels, n_filters, (filter_size)),
			# nn.MaxPool1d((pool_filter_size)), # Max pool layers, optional. 
			nn.Conv1d(n_filters, n_filters, (filter_size)),
			# nn.MaxPool1d((pool_filter_size)),
			nn.Conv1d(n_filters, n_filters, (filter_size)),
			nn.Conv1d(n_filters, n_filters, (filter_size))
			)

		# LSTM layers
		self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)

		# Dropout layer
		self.dropout2d = nn.Dropout2d(p=drop_prob)
		self.dropout = nn.Dropout(p=drop_prob)

		# Output layer
		self.predictor = nn.Linear(n_hidden,n_classes)

	
	def forward(self, x, hidden, batch_size):

		#Reshape x if necessary to add the 2nd dimension
		x = x.view(-1, n_channels, len_seq)
		# print(x.size())
		x = self.convlayer(x)
		x = x.view(batch_size, -1, self.n_filters)
		

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



def train(net, X_train, y_train,X_val,y_val, epochs=num_epochs, batch_size=batch_size, lr=lr, time=True, shuffle=True):


	if time:
		print('Starting training at',datetime.now())
		start_time=datetime.now()

	weight_decay = 1e-5*lr*batch_size*(50/batchlen)
	opt = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
	scheduler = torch.optim.lr_scheduler.StepLR(opt,lr_step) # Learning rate scheduler to reduce LR every 100 epochs

	if(train_on_gpu):
		net.cuda()

	train_stats = np.unique([a for y in y_train for a in y],return_counts=True)[1]
	val_stats = np.unique([a for y in y_val for a in y],return_counts=True)[1]

	print('Training set statistics:')
	print(len(train_stats),'classes with distribution',train_stats)
	print('Validation set statistics:')
	print(len(val_stats),'classes with distribution',val_stats)
	
	criterion = nn.CrossEntropyLoss(weight=torch.tensor([max(train_stats)/i for i in train_stats],dtype=torch.float).cuda()) # Prepare weighted cross entropy for training and validation.
	val_criterion = nn.CrossEntropyLoss()

	early_stopping = EarlyStopping(patience=patience, verbose=False)

	with open('log.csv', 'w', newline='') as csvfile: # We will save some training statistics to plot a loss curve later.

		for e in range(epochs):
			
			train_losses = []
			net.train() # Setup network for training



			for batch in iterate_minibatches_2D(X_train, y_train, batch_size, len_seq, stride, shuffle=shuffle, num_batches=num_batches, oversample=False, batchlen=batchlen, val=True):

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
				for batch in iterate_minibatches_2D(X_val, y_val, val_batch_size, len_seq, stride, shuffle=shuffle, num_batches=num_batches_val, batchlen=batchlen, val=True):
					

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



def test(net, X_test, y_test, batch_size, remove_nulls=False, shuffle=True):

	print('Starting testing at', datetime.now())
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
			
		for batch in iterate_minibatches_2D(X_test, y_test, test_batch_size, len_seq, stride, shuffle=True, num_batches=-1, batchlen=batchlen, val=True):
				
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




	print('Finished testing at', datetime.now())
	print('Total time elapsed during inference:', (datetime.now()-start_time).total_seconds())

	f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')

	classreport = classification_report(targets_cumulative, top_classes,target_names=opp_class_names)
	confmatrix = confusion_matrix(targets_cumulative, top_classes,normalize='true')
	print('---- TESTING REPORT ----')
	print(classreport)

	df_cm = pd.DataFrame(confmatrix, index=opp_class_names,columns=opp_class_names)
	plt.figure(10,figsize=(15,12))
	sn.heatmap(df_cm,annot=True,fmt='.2f',cmap='YlOrRd')
	plt.savefig('Results/{:.2f}_Confmatrix_{}_{}_{}_{}.png'.format(f1score,datetime.now(),lr,batchlen,batch_size))
	plt.show(plt.figure(10))



if __name__ == '__main__':

	print('==HYPERPARAMETERS==')
	print('learning rate',lr,'batch length',batchlen,'batch size',batch_size)

	X_train,y_train = load_opp_runs('train',len_seq, stride)
	X_val, y_val = load_opp_runs('val',len_seq, stride)

	net = DeepConvLSTM()

	net.apply(init_weights)

	train_on_gpu = torch.cuda.is_available() # Check for cuda

	try:
		train(net,X_train,y_train,X_val,y_val, batch_size=batch_size, lr=lr)
	except(KeyboardInterrupt):
		pass

	del X_train
	del y_train
	del X_val
	del y_val

	plot_data(save_fig='{}-{}-{}-{}'.format(datetime.now(),lr,batchlen,batch_size))

	print('Loading test data')

	X_test, y_test = load_opp_runs('test',len_seq, stride)

	print('Testing fully trained model.')

	test(net, X_test,y_test, batch_size=batch_size, remove_nulls=False)

	torch.save(net.state_dict(), 'fullytrained.pt')

	print('Loading model state_dict from checkpoint.')

	state_dict = torch.load('checkpoint.pt')

	net.load_state_dict(state_dict)

	print('Testing checkpointed model.')

	test(net, X_test,y_test, batch_size=batch_size, remove_nulls=False)

	print('Plotting training curves')

