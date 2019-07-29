import pandas as pd
from collections import Counter
import os
import numpy as np
import socket
import pdb

np.random.seed(2017)
RAW_DATA = '../raw_data'
RATINGS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.data')
RATINGS = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
USERS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.user')
USERS = pd.read_csv(USERS_FILE, sep='|', header=None)
ITEMS_FILE = os.path.join(RAW_DATA, 'ml-100k/item.csv')
ITEMS = pd.read_csv(ITEMS_FILE, header=None, encoding="ISO-8859-1")
OUT_DIR = '../dataset/'


def format_user_file(user_df):
    formatted = user_df[[0, 1, 2, 3]].copy()

    min_age, max_age = 15, 55
    formatted[1] = formatted[1].apply(lambda x: max_age if x > max_age else x)
    formatted[1] = formatted[1].apply(lambda x: min_age if x < min_age else x)
    formatted[1] = formatted[1].apply(lambda x: max_age / 5 if x >= max_age else min_age / 5 if x <= min_age else x / 5)
    # print Counter(formatted[1])
    formatted[1] = formatted[1].apply(lambda x: int(x - formatted[1].min()))
    formatted[2] = formatted[2].apply(lambda x: {'M': 0, 'F': 1}[x])
    occupation = dict(
        [(o.strip(), i) for i, o in enumerate(open(os.path.join(RAW_DATA, 'ml-100k/u.occupation'), 'r').readlines())])
    formatted[3] = formatted[3].apply(lambda x: occupation[x])
    formatted = formatted.fillna(-1)
    formatted.columns = ['uid', 'u_age', 'u_gender', 'u_occupation']
    # print formatted
    # print formatted.info()
    return formatted


def format_item_file(item_df):
    formatted = item_df.drop([1, 3, 4], axis=1).copy()
    #formatted = item_df.drop([1, 3, 4, 24, 25], axis=1).copy()
    formatted.columns = range(len(formatted.columns))
    formatted[1] = formatted[1].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else -1)
    min_year = 1989
    formatted[1] = formatted[1].apply(lambda x: min_year if 0 < x < min_year else x)
    formatted[1] = formatted[1].apply(lambda x: min_year + 1 if min_year < x < min_year + 4 else x)
    years = dict([(year, i) for i, year in enumerate(sorted(Counter(formatted[1]).keys()))])
    formatted[1] = formatted[1].apply(lambda x: years[x])
    formatted.columns = ['iid', 'i_year',
                         'i_Action', 'i_Adventure', 'i_Animation', "i_Children's", 'i_Comedy',
                         'i_Crime', 'i_Documentary ', 'i_Drama ', 'i_Fantasy ', 'i_Film-Noir ',
                         'i_Horror ', 'i_Musical ', 'i_Mystery ', 'i_Romance ', 'i_Sci-Fi ',
    #                     'i_Thriller ', 'i_War ', 'i_Western', 'i_Other']
                         'i_Thriller ', 'i_War ', 'i_Western', 'i_Other', 'i_Actor', 'i_Director']
    # print Counter(formatted[1])
    # print formatted
    # print formatted.info()
    return formatted


def format_rating(ratings, users, items):
    ratings = ratings.drop(3, axis=1).copy()
    ratings.columns = ['uid', 'iid', 'rating']
    ratings = pd.merge(ratings, users, on='uid', how='left')
    ratings = pd.merge(ratings, items, on='iid', how='left')
    # print ratings
    return ratings


def random_split_data():
    dir_name = 'ml-100k-r'
    if not os.path.exists(os.path.join(OUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUT_DIR, dir_name))
    users = format_user_file(USERS)
    users.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.users.csv'), index=False)
    items = format_item_file(ITEMS)
    items.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.items.csv'), index=False)
    all_data = format_rating(RATINGS, users, items)
    all_data.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.all.csv'), index=False)

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    train_size = int(len(all_data) * 0.8)
    validation_size = int(len(all_data) * 0.1)
    train_set = all_data[:train_size]
    validation_set = all_data[train_size:train_size + validation_size]
    test_set = all_data[train_size + validation_size:]
    # print train_set
    # print validation_set
    # print test_set
    train_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.train.csv'), index=False)
    validation_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.validation.csv'), index=False)
    test_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.test.csv'), index=False)

def main():
    random_split_data()

if __name__ == '__main__':
    main()
