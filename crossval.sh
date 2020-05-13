for sub in 1 2 3 4
do
	echo 'Leaving out subject' $sub 'for testing'
	python3 preprocess_data.py -i $1 -s $sub
	python3 -u DeepConvLSTM_py3_v0-6.py | tee log_$sub.txt 
done