import sys
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import recommender as r

# Nlp
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
#import spacy
#nlp = spacy.load('en_core_web_sm')

import warnings
warnings.filterwarnings('ignore')

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import dask
import dask.dataframe as dd
from dask.distributed import Client, progress

def user_analysis():
    dtype_users = {'city': np.int32,
                   'bd': np.int32,
                   'registered_via': np.int32,
                   'registration_init_time': np.int32,
                   'expiration_date': np.int32}

    df_users = pd.read_csv('members.csv', dtype=dtype_users)

    # parse date calumns
    for i in ['registration_init_time', 'expiration_date']:
        df_users[i] = pd.to_datetime(df_users[i], format='%Y%m%d',
                                     errors='ignore')

    df_users['registration_year'] = (
        pd.DatetimeIndex(df_users['registration_init_time']).year
    )


    # find all mean age
    warnings.filterwarnings('ignore')

    set_registr_year = set(df_users.registration_year.values)
    set_registr_via = set(df_users.registered_via.values)

    dict_fill_age = {}
    for i in set_registr_year:
        for j in set_registr_via:
            try:
                dict_fill_age[tuple([i, j])] = (
                    int(
                        df_users[(df_users.bd >= 13)
                        & (df_users.bd <= 80)][(df_users.registration_year == i)
                        & (df_users.registered_via == j)].bd.mean()
                    )
                )
            except ValueError:
                pass

    # now fill all values that are not in (13, 80) with the mean age based
    # on the year and the method of registration
    def fill_age(df_users):
        for key, val in dict_fill_age.items():
            c = np.logical_and(
                    np.logical_and(
                        df_users.registration_year == key[0],
                        df_users.registered_via == key[1]
                    ),
                    np.logical_or(df_users.bd < 13, df_users.bd > 80)
                )

            if c:
                return val
            elif np.logical_and(df_users.bd >= 13, df_users.bd <= 80):
                return df_users.bd
            else:
                pass

    df_users.bd = df_users.apply(fill_age, axis=1)

    x = df_users[['bd', 'registration_year', 'registered_via']]

    # I'll fill missing age with the overall mean
    df_users.bd.fillna(df_users.bd.mean(), inplace=True)
    df_users.bd = df_users.bd.astype('int32')

    del x

    df_users.drop(labels=['gender',
                          'registration_init_time',
                          'expiration_date',
                          'registration_year'], axis=1, inplace=True)

    return df_users


df_users = user_analysis()
df_users.head()


for df in pd.read_csv('items2.csv', chunksize=100_000):
    display(df.head(3))
    break

###########
###########
###########

client = Client('tcp://192.168.1.10:8786')
client.restart()
#client.get_versions(check=True)

def read_fromS3(file_path):
    aws_ids = pd.read_csv('aws_ids.csv', header=1)
    AWSAccessKeyId = aws_ids.AWSAccessKeyId[0]
    AWSSecretKey = aws_ids.AWSSecretKey[0]

    df_interactions = dd.read_csv(file_path,
                                  storage_options= {'key': AWSAccessKeyId,
                                                    'secret': AWSSecretKey})
    return df_interactions
#df_interactions = read_fromS3('s3://stock-mdh-datascience/train.csv')

def construct_pandas_df():
    dtype_interactions = {'date': np.int16,
                          'interacted': np.int16}

    df_inter = pd.DataFrame()
    for df in pd.read_csv('train.csv', chunksize=100_000, dtype=dtype_interactions):
        df_inter = df_inter.append(df)

    return df_inter
df_inter = construct_pandas_df()

%%time
df_interactions = dd.from_pandas(df_inter, npartitions=16)

# One thing to know from the doc
# (http://docs.dask.org/en/latest/dataframe-performance.html)
# Often DataFrame workloads look like the following:

# - Load data from files
# - Filter data to a particular subset
# - Shuffle data to set an intelligent index
# - Several complex queries on top of this indexed data
#
# It is often ideal to load, filter, and shuffle data once and keep this result
# in memory. Afterwards, each of the several complex queries can be based off of
# this in-memory data rather than have to repeat the full load-filter-shuffle
# process each time. To do this, use the client.persist method

# We import data from S3 (or from pandas) in Xseconds. It will be a great choice
# to use client.persist method in order to avoid this execution time each time
# we want to process our data.

user_item = df_interactions[['msno', 'song_id', 'interacted']]
user_item

# persist data across the cluster (computation done in the background, see
# diagnostic)
user_item = client.persist(user_item, retries=2)

# suggestion to convert to cat by Rocklin
user_item = user_item.categorize(columns=['song_id'])
user_item = client.persist(user_item, retries=2)
user_item



user_item_df = user_item.pivot_table(index='msno',
                                     columns='song_id',
                                     values='interacted',
                                     aggfunc='mean')


user_item_df.npartitions

# reset nb partitions to 16
user_item_df = user_item_df.repartition(npartitions=user_item_df.npartitions*16)
user_item_df = client.persist(user_item_df, retries=2)


#user_item_df = user_item_df.repartition(npartitions=user_item_df.npartitions * 10)
#user_item_df.npartitions

%%time
#with dask.config.set(shuffle='tasks'):
    #user_item_df.head()


# read all the data (already done for df interactions)
dtype_items = {'song_length': np.int32,
               'language': np.float32}

df_items = pd.read_csv('items2.csv', dtype=dtype_items)
























rec = r.Recommender(df_items=df_items,
                    df_reviews=df_interactions,
                    user_item_df=user_item_df,
                    item_name_colname='name',
                    user_id_colname='msno',
                    item_id_colname='song_id',
                    rating_col_name='interacted',
                    date_col_name='date')

rec.fit(iters=5)

df_user_similarity = user_item_df.reset_index().replace(np.nan, 0)
def prep_get_similar_user():
    user_content = np.array(df_user_similarity.iloc[:,1:])
    user_content_transpose = np.transpose(user_content)
    dot_prod = user_content.dot(user_content_transpose)
    return dot_prod

dot_product_matrix_user = prep_get_similar_user()

temp_df_item_similarity = df_items.iloc[:,3:]

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

item_content = mms.fit_transform(temp_df_item_similarity)

del temp_df_item_similarity

# Maybe use dask here to go faster (will try later)

def prep_get_similar_item():
    item_content_transpose = np.transpose(item_content)
    dot_prod = item_content.dot(item_content_transpose)
    return dot_prod

dot_product_matrix_item = prep_get_similar_item()

def display_recommendations(rec_ids, rec_names, message, rec_ids_users, rec_user_articles):

    if type(rec_ids) == type(None):
        print(f"{message}")

    else:
        dict_id_name = dict(zip(rec_ids, rec_names))

        if type(rec_ids_users) != type(None):
            print('Matrix Factorisation SVD:')
            print(f"\t{message}")

            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")

            print('CF User Based:')
            print('\tUser that are similar to you also enjoy:\n')
            for i in rec_user_articles[:5]:
                print(f"\t- {i}")
        else:
            print(f"\t{message}")
            dict_id_name = dict(zip(rec_ids, rec_names))
            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")


list_existing_user_ids = rec.user_ids_series
list_existing_item_ids = rec.items_ids_series

rec_ids, rec_names, message, rec_ids_users, rec_user_articles = rec.make_recommendations(_id=3,
                                                                                         dot_prod_user= dot_product_matrix_user,
                                                                                         dot_prod_item=dot_product_matrix_item,
                                                                                         _id_type='user',
                                                                                         rec_num=5)
display_recommendations(rec_ids, rec_names, message, rec_ids_users, rec_user_articles)
