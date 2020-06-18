# DeepConvLSTM_py3
DeepConvLSTM implemented in python 3 and pytorch

Included is an implementation of DeepConvLSTM in python 3 and torch, along with a jupyter notebook giving an example of training and testing the model on the Opportunity challenge dataset.


### Opportunity example notebook
To interact with the notebook, you can use pipenv (https://pypi.org/project/pipenv/) to create a virtual environment with all of the requirements in the included pipfile installed. 

After installing pipenv, just run

```
pipenv install
```

in the directory containing the pipfile, then 

```
pipenv run jupyter notebook
```

and choose opportunity_example.ipynb. 

### Main script

After installing all dependencies with pipenv as above you will be able to run the main DeepConvLSTM_py3.py script. Use

```
pipenv run DeepConvLSTM_py3.py -h
```

for a list of command line options.



