__author__ = 'fjordonez'

import os
import zipfile
import argparse
import numpy as np
import _pickle as cp

from io import BytesIO
from pandas import Series


# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

def select_subject(test):

    train = ['1','2','3','4']
    runs = ['Drill','ADL1','ADL2','ADL3','ADL4']
    val_runs = ['ADL4']



    test_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(test,run) for run in runs]

    train.remove(str(test))
    runs.remove(val_runs[0])

    train_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(sub,run) for sub in train for run in runs]
    val_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(sub,run) for sub in train for run in val_runs]

    return train_files, test_files, val_files




# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   
                       10000,  10000,
                       10000,  10000,  10000,  10000 ,  250 ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                        -10000, -10000,
                       -10000, -10000, -10000, -10000 ,-250 ]


def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    ## ACC/GYRO ONLY
    # features_delete = np.arange(43, 50) #Exclude quats and magnetometer reading from BACK
    # features_delete = np.concatenate([features_delete, np.arange(56, 63)]) #Exclude quats and magnetometer reading from RUA
    # features_delete = np.concatenate([features_delete, np.arange(69, 76)]) #Exclude quats and magnetometer reading from RLA
    # features_delete = np.concatenate([features_delete, np.arange(82, 89)]) #Exclude quats and magnetometer reading from LUA
    # features_delete = np.concatenate([features_delete, np.arange(95, 134)]) #Exclude quats and magnetometer reading from LLA and shoes
    # features_delete = np.concatenate([features_delete, np.arange(134, 243)]) #Exclude ambient item sensors
    # features_delete = np.concatenate([features_delete, np.arange(244, 249)])

    ## ACC/GYRO/MAG/QUAT ONLY
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 1:NB_SENSOR_CHANNELS+1]


    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, NB_SENSOR_CHANNELS+1]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, NB_SENSOR_CHANNELS+2]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    """Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location

    :param data_set:
            Path with original OPPORTUNITY zip file
    :return:
    """
    print('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print('... dataset path {0} not found'.format(data_set))
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y =  divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(dataset, test_sub, label):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_dir = check_data(dataset)

    train_files, test_files, val_files = select_subject(test_sub)

    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')

    try:
        os.mkdir('data')
    except FileExistsError:
        pass


    # Generate training files
    print('Generating training files')
    for i,filename in enumerate(train_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> train_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label)
            with open('data/train_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    # Generate validation files
    print('Generating validation files')
    for i,filename in enumerate(val_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> val_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label)
            with open('data/val_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    # Generate testing files
    print('Generating testing files')
    for i,filename in enumerate(test_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> test_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label)
            with open('data/test_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))




def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    # parser.add_argument(
    #     '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-s','--subject', type=int, help='Subject to leave out for testing', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized', default="gestures", choices = ["gestures", "locomotion"], required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = 'data/raw/OpportunityUCIDataset.zip'
    subject = args.subject
    label = args.task
    # Return all variable values
    return dataset, subject, label

if __name__ == '__main__':

    OpportunityUCIDataset_zip, sub, l = get_args();
    generate_data(OpportunityUCIDataset_zip, sub, l)
