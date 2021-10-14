import os
import zipfile
import argparse
import numpy as np

from io import BytesIO
from pandas import Series
import pandas as pd

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from utils import sliding_window, paint, plot_pie
else:
    # uses current package visibility
    from .utils import sliding_window, paint, plot_pie

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113


def select_subject(dataset_name):
    # Test set for the opportunity challenge.
    train_runs = ['S1-Drill', 'S1-ADL1', 'S1-ADL2', 'S1-ADL3', 'S1-ADL4', 'S2-Drill', 'S2-ADL1', 'S2-ADL2',
                  'S3-Drill', 'S3-ADL1', 'S3-ADL2', 'S2-ADL3', 'S3-ADL3']
    val_runs = ['S1-ADL5']
    test_runs = ['S2-ADL4', 'S2-ADL5', 'S3-ADL4', 'S3-ADL5']

    train_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in train_runs]
    val_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in val_runs]
    test_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in test_runs]

    return train_files, test_files, val_files

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


def normalize(data, mean, std):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param mean: numpy integer array
        Array containing mean values for each sensor channel
    :param std: numpy integer array
        Array containing the standard deviation of each sensor channel
    :return:
        Normalized sensor data
    """
    return (data - mean) / std


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    if label in ['locomotion', 'gestures']:
        data_x = data[:, 1:NB_SENSOR_CHANNELS + 1]
        if label == 'locomotion':
            data_y = data[:, NB_SENSOR_CHANNELS + 1]  # Locomotion label
        elif label == 'gestures':
            data_y = data[:, NB_SENSOR_CHANNELS + 2]  # Gestures label

    elif label == -1:

        data_x = data[:, 1:-1]
        data_y = data[:, -1]


    else:
        raise RuntimeError("Invalid label: '%s'" % label)

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


def process_dataset_file(dataset_name, data, label):
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
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Replace trailing NaN values with 0.0
    data_x = pd.DataFrame(data_x)
    for column in data_x:
        ind = data_x[column].last_valid_index()
        data_x[column][ind:] = data_x[column][ind:].fillna(0.0)
    data_x = data_x.to_numpy()
    
    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    return data_x, data_y


def generate_data(dataset, dataset_name, label):
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

    train_files, test_files, val_files = select_subject(dataset_name)

    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')

    try:
        os.mkdir('data/opportunity')
    except FileExistsError:  # Remove data if already there.
        for file in os.scandir('data/opportunity'):
            if 'data' in file.name:
                os.remove(file.path)

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty(0, dtype=np.uint8)

    # Generate training files
    print('Generating training files')
    for i, filename in enumerate(train_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> train_data'.format(filename))
            x, y = process_dataset_file(dataset_name, data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    mean_train = np.mean(data_x, axis=0)
    std_train = np.std(data_x, axis=0)

    data_x = normalize(data_x, mean_train, std_train)

    np.savez_compressed('data/opportunity/train_data.npz', data=data_x, target=data_y)
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty(0, dtype=np.uint8)
    # Generate validation files
    print('Generating validation files')
    for i, filename in enumerate(val_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> val_data'.format(filename))
            x, y = process_dataset_file(dataset_name, data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    data_x = normalize(data_x, mean_train, std_train)

    np.savez_compressed('data/opportunity/val_data.npz', data=data_x, target=data_y)
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty(0, dtype=np.uint8)
    # Generate testing files
    print('Generating testing files')
    for i, filename in enumerate(test_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> test_data'.format(filename))
            x, y = process_dataset_file(dataset_name, data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    data_x = normalize(data_x, mean_train, std_train)

    np.savez_compressed(f'data/opportunity/test_data.npz', data=data_x, target=data_y)


def partition(path, window, stride):

    # read raw datasets (sample-level)
    print(f"[*] Reading raw files from {path}")
    dataset_train = np.load(os.path.join(path, "train_data.npz"))
    x_train, y_train = dataset_train["data"], dataset_train["target"]
    dataset_val = np.load(os.path.join(path, "val_data.npz"))
    x_val, y_val = dataset_val["data"], dataset_val["target"]
    dataset_test = np.load(os.path.join(path, "test_data.npz"))
    x_test, y_test = dataset_test["data"], dataset_test["target"]

    # apply sliding window over raw samples and generate segments
    data_train, target_train = sliding_window(x_train, y_train, window, stride)
    data_val, target_val = sliding_window(x_val, y_val, window, stride)
    data_test, target_test = sliding_window(x_test, y_test, window, stride)
    data_test_sample_wise, target_test_sample_wise = sliding_window(
        x_test, y_test, window, 1
    )

    # show processed datasets info (segment-level)
    print(
        "[-] Train data : {} {}, target {} {}".format(
            data_train.shape, data_train.dtype, target_train.shape, target_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            data_val.shape, data_val.dtype, target_val.shape, target_val.dtype
        )
    )
    print(
        "[-] Test data : {} {}, target {} {}".format(
            data_test.shape, data_test.dtype, target_test.shape, target_test.dtype
        )
    )
    print(
        "[-] Test data sample-wise : {} {}, target sample-wise {} {}".format(
            data_test_sample_wise.shape,
            data_test_sample_wise.dtype,
            target_test_sample_wise.shape,
            target_test_sample_wise.dtype,
        )
    )

    # save processed datasets (segment-level)
    np.savez_compressed(
        os.path.join(path, "train_data.npz"), data=data_train, target=target_train
    )
    np.savez_compressed(
        os.path.join(path, "val_data.npz"), data=data_val, target=target_val
    )
    np.savez_compressed(
        os.path.join(path, "test_data.npz"), data=data_test, target=target_test
    )
    np.savez_compressed(
        os.path.join(path, "test_sample_wise.npz"),
        data=data_test_sample_wise,
        target=target_test_sample_wise,
    )
    print("[+] Processed segment datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def find_data(name):
    dataset_dir = 'data/raw/'
    dataset_names = {'opportunity': 'OpportunityUCIDataset.zip'}
    dataset = dataset_dir + dataset_names[name]

    return dataset


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    # parser.add_argument(
    #     '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-s', '--subject', type=str, help='Subject to leave out for testing', required=False, default='test')
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized (for opportunity)',
        default="gestures", choices=["gestures", "locomotion"], required=False)
    parser.add_argument(
        '-w', '--window_size', type=int, help='Size of sliding window (in samples). Default = 24',
        default=24, required=False)
    parser.add_argument(
        '-ws', '--window_step', type=int, help='Stride of sliding window. Default = 12',
        default=12, required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    subject = args.subject
    label = args.task
    # Return all variable values
    return subject, label, args


if __name__ == '__main__':
    dataset_name = 'opportunity'
    sub, l, args = get_args()
    dataset = find_data(dataset_name)
    generate_data(dataset, sub, l)
    partition(f'data/opportunity', args.window_size, args.window_step)
