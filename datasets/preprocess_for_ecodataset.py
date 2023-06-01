from absl import app, flags
import numpy as np
import pandas as pd
from datasets.tfrecord import write_tfrecord
FLAGS = flags.FLAGS
flags.DEFINE_integer("household_id", 2, "Indicator of household data")
# TODO: Understand flags, app

def main(argv):
    # 1. Load data
    X = pd.read_csv(f"./datasets/{FLAGS.household_id}_X_train.csv").values
    y = pd.read_csv(f"./datasets/{FLAGS.household_id}_Y_train.csv").values.reshape(-1)
    
    # 2. Apply Sliding Window
    len_data, H = X.shape
    filter_len = 3
    N = len_data - filter_len + 1
    filtered_X = np.zeros((N, filter_len, H))
    for i in range(0, N):
        # print(f"(Start, End) = {i, i+filter_len-1}")
        start = i
        end = i+filter_len
        filtered_X[i] = X[start:end]
    filtered_y = y[filter_len-1:]

    # 3. Split into train, val, test
    len_partition = filtered_X.shape[0] // 3

    train_X = filtered_X[:len_partition].tolist()
    val_X = filtered_X[len_partition:len_partition*2].tolist()
    test_X = filtered_X[len_partition*2:].tolist()

    train_y = filtered_y[:len_partition].tolist()
    val_y = filtered_y[len_partition:len_partition*2].tolist()
    test_y = filtered_y[len_partition*2:].tolist()

    # 4. Write tfrecord
    write_tfrecord(f"./datasets/tfrecords/ecodataset_{FLAGS.household_id}_train.tfrecord", train_X, train_y)
    write_tfrecord(f"./datasets/tfrecords/ecodataset_{FLAGS.household_id}_val.tfrecord", val_X, val_y)
    write_tfrecord(f"./datasets/tfrecords/ecodataset_{FLAGS.household_id}_test.tfrecord", test_X, test_y)

if __name__ == "__main__":
    app.run(main)
