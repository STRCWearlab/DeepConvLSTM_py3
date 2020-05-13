

for sub in 1 2 3 4
do
	echo 'Leaving out subject' $sub 'for testing'
	python3 src/data/preprocess_data.py -i '/home/lloyd/Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch/data/raw/OpportunityUCIDataset.zip' -s $sub
	python3 -u DeepConvLSTM_py3_v0-6.py | tee log_$sub.txt 
done