import numpy as np
import pandas as pd
import sys
import random
import os
import argparse
import utils.utils as utils

np.random.seed(43)
random.seed(43)
np.set_printoptions(threshold=sys.maxsize)


def main(args):

    for dataset in args.datasets:
        print("Working on", dataset, "dataset")

        # Read the dataset and prepare it for training, validation and test
        names = ['user_id', 'item_id', 'rating', 'utc']
        if args.parse_dates:
            parse_dates = ['utc']
            df = pd.read_csv('raw_datasets/' + dataset + '.tsv', sep='\t',
                             dtype={'rating': 'float64', 'utc': 'str'}, parse_dates=parse_dates, header=0, names=names)
        else:
            df = pd.read_csv('raw_datasets/' + dataset + '.tsv', sep='\t',
                             dtype={'rating': 'float64', 'utc': 'int64'}, header=0, names=names)
        df = df.groupby('user_id').filter(lambda x: len(x) >= 20)
        df = df.groupby(['user_id', 'item_id'])['utc'].max().reset_index()
        print(df.shape[0], 'interactions read')
        df, _ = utils.convert_unique_idx(df, 'user_id')
        df, _ = utils.convert_unique_idx(df, 'item_id')
        user_size = len(df['user_id'].unique())
        item_size = len(df['item_id'].unique())
        print('Found {} users and {} items'.format(user_size, item_size))
        total_user_lists = utils.create_user_lists(df, user_size, 4)
        train_user_lists, validation_user_lists, test_user_lists = utils.split_train_test(total_user_lists,
                                                                                          test_size=0.2)

        if not os.path.exists('sets'):
            os.makedirs('sets')
        with open('datasets/{}_trainingset.tsv'.format(dataset), 'w') as out:
            for u, train_list in enumerate(train_user_lists):
                for i in train_list:
                    out.write(str(u) + '\t' + str(i) + '\t' + str(1) + '\n')
        with open('datasets/{}_testset.tsv'.format(dataset), 'w') as out:
            for u, test_list in enumerate(test_user_lists):
                for i in test_list:
                    out.write(str(u) + '\t' + str(i) + '\t' + str(1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='Set the datasets you want to use', required=True)
    parser.add_argument('--parse_dates', action='store_true', help='Set if UTC contains dates')
    parsed_args = parser.parse_args()
    main(parsed_args)
