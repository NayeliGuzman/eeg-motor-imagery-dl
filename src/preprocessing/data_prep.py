import numpy as np
from tensorflow.keras.utils import to_categorical

# ----------------------------
# Label adjustment
# ----------------------------
def adjust_labels(y_train_valid, y_test):
    y_train_valid = y_train_valid - 769
    y_test = y_test - 769
    return y_train_valid, y_test


# ----------------------------
# Train/val splitting
# ----------------------------
def train_valid_split(X, y, valid_size=375):
    ind_valid = np.random.choice(len(X), valid_size, replace=False)
    ind_train = np.array(list(set(range(len(X))) - set(ind_valid)))

    return (X[ind_train], X[ind_valid],
            y[ind_train], y[ind_valid])


# ----------------------------
# preprocessing
# ----------------------------

def data_prep(X,y,sub_sample,average,noise):

    total_X = None
    total_y = None

    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    print('Shape of X after trimming:',X.shape)

    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)


    total_X = X_max
    total_y = y
    #reduce input size for computational efficiency and to extract more abstract features.
    print('Shape of X after maxpooling:',total_X.shape)

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    #This is dataset-level augmentation — doesn’t affect the per-sample dimensions (22, 250), only increases the number of samples.
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)

    # Subsampling

    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))


    print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y

# ----------------------------
# One-hot encoding 
# ----------------------------
def encode_labels(y, num_classes=4):
    return to_categorical(y, num_classes)

# ----------------------------
# Reshaping
# ----------------------------
def format_data(x_train, x_valid, x_test, mode="cvae"):
    """
    mode:
        - "cvae" → (22, 250, 1)
        - "cnn"  → CNN-style reshape
    """
    # Reshaping the training and validation dataset
    # For VAE we do not need swapaxes calls or cnn style reshaping
    if mode == "cvae":
        # #Add singleton channel dimension for Conv2D
        # x_train_cvae = x_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        # x_valid_cvae = x_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
        # x_test_cvae  = x_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
        # # x_train_cvae.shape -> (samples, 22, 250, 1)

        # x_train_cvae = x_train   # already (6960, 22, 250, 1)
        # x_valid_cvae = x_valid
        # x_test_cvae  = x_test

        print('Training shape for CVAE:', x_train.shape)
        print('Validation shape for CVAE:', x_valid.shape)
        print('Test shape for CVAE:', x_test.shape)

    elif mode == "cnn":
        x_train = np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 1, 2)

        x_valid = np.swapaxes(x_valid, 1, 3)
        x_valid = np.swapaxes(x_valid, 1, 2)

        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 1, 2)

        print('Shape of training set after dimension reshaping:',x_train.shape)
        print('Shape of validation set after dimension reshaping:',x_valid.shape)
        print('Shape of test set after dimension reshaping:',x_test.shape)
        
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return x_train, x_valid, x_test


# ----------------------------
# Full Pipeline
# ----------------------------
def prepare_dataset(
    X_train_valid,
    y_train_valid,
    X_test,
    y_test,
    valid_size=375,
    sub_sample=2,
    average=2,
    noise=True,
    num_classes=4,
    mode="cvae"
    ):
    """
    Full preprocessing pipeline from raw EEG → model-ready tensors
    """
    print("RUNNING NEW VERSION OF prepare_dataset")
    ind_valid = np.random.choice(len(X_train_valid), valid_size, replace=False) # len(X_train_valid)=2115
    ind_train = np.array(list(set(range(len(X_train_valid))).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (X_train, X_valid) = X_train_valid[ind_train], X_train_valid[ind_valid]
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]


    # Augmenting the data
    x_train,y_train = data_prep(X_train,y_train,2,2,True)
    x_valid,y_valid = data_prep(X_valid,y_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)


    print('Shape of testing set:',X_test_prep.shape)
    print('Shape of testing labels:',y_test_prep.shape)

    print('Shape of training set:',x_train.shape)
    print('Shape of validation set:',x_valid.shape)
    print('Shape of training labels:',y_train.shape)
    print('Shape of validation labels:',y_valid.shape)

    # Converting the labels to categorical variables for multiclass classification
    y_train = encode_labels(y_train, 4)
    y_valid = encode_labels(y_valid, 4)
    y_test = encode_labels(y_test_prep, 4)
    print('Shape of training labels after categorical conversion:',y_train.shape)
    print('Shape of validation labels after categorical conversion:',y_valid.shape)
    print('Shape of test labels after categorical conversion:',y_test.shape)

    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
    print('Shape of training set after adding width info:',x_train.shape)
    print('Shape of validation set after adding width info:',x_valid.shape)
    print('Shape of test set after adding width info:',x_test.shape)

    x_train, x_valid, x_test = format_data(x_train, x_valid, x_test, mode)


    return x_train, x_valid, x_test, y_train, y_valid, y_test
