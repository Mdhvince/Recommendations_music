import sys
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import recommender1by1 as robo
import warnings
warnings.filterwarnings('ignore')
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

SCHEDULER = 'tcp://192.168.1.10:8786'
client = Client(SCHEDULER)
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

def construct_pandas_df():
    dtype_interactions = {'date': np.int16,
                          'interacted': np.int16}

    df_inter = pd.DataFrame()
    for df in pd.read_csv('train.csv', chunksize=100_000, dtype=dtype_interactions):
        df_inter = df_inter.append(df)

    return df_inter
df_inter = construct_pandas_df()
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
user_item = client.persist(user_item, retries=2)
user_item_grouped = user_item.groupby(['msno',
                                       'song_id'])['interacted'].max()

user_item_grouped_pd = client.compute(user_item_grouped).result()
type(user_item_grouped_pd)


dtype_items = {'song_length': np.int32,
               'language': np.float32}

df_items = pd.read_csv('items2.csv', dtype=dtype_items)
df_inter['date'] = 0


#MAKE RECOMMENDATIONS
rec = robo.Recommender(df_items=df_items, df_reviews=df_inter,
                      user_item_grouped=user_item_grouped_pd,
                      item_name_colname='name', user_id_colname='msno',
                      item_id_colname='song_id', rating_col_name='interacted',
                      date_col_name='date')



list_existing_user_ids = rec.user_item_grouped.reset_index()[rec.user_id_colname].unique()
list_existing_user_ids[0]



rec_ids, rec_names, message = rec.make_recommendations(_id=list_existing_user_ids[0], _id_type='user',rec_num=5)

message
rec_ids
rec_names
