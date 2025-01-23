''' 
Demo script for preprocessing adjusted from:

https://github.com/data-science-on-aws/data-science-on-aws/blob/oreilly-book/06_prepare/preprocess-scikit-text-to-bert-feature-store.py#L394
''' 
import subprocess
import sys
# This is 2.3.0 (vs. 2.3.1 everywhere else) because we need to
# use anaconda and anaconda only supports 2.3.0 at this time
subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "anaconda", "tensorflow==2.3.0", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.3.1"])

import tensorflow as tf

import argparse
import pandas as pd
import collections

from sklearn.preprocessing import MinMaxScaler    


def clean_data(input_data):
    df = pd.read_csv(input_data)
    categories = {c: i for i, c in enumerate(set(df.Product_Category))}

    df.dropna(subset=['Date'], inplace=True)
    df.dropna(subset=['Order_Demand'], inplace=True)

    df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
    df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
    df['Order_Demand'] = df['Order_Demand'].astype('int64')

    df['Category'] = df.Product_Category.apply(lambda x: categories[x])
    df['Date'] = pd.to_datetime(df['Date']) 
    df['Weeks']  = df['Date'].dt.to_period('W').dt.to_timestamp()

    condition = (df.Warehouse == 'Whse_A') & (df.Date > '2012-01-01')
    grain = ['Weeks', 'Category']
    fields = ['Order_Demand'] + grain

    weekly = df[condition][fields].groupby(grain).sum().reset_index().sort_values('Weeks')
    data_pivot = weekly.pivot(index='Weeks', columns='Category', values='Order_Demand').fillna(0)
    return data_pivot, categories


def scale_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

    
def write(scaled_data, output_file):
    records = []
    tf_record_writer = tf.io.TFRecordWriter(output_file)
    for _, x in enumerate(scaled_data):
        all_features = collections.OrderedDict()
        all_features["demand"] = tf.train.Feature(float_list=tf.train.FloatList(value=list(x)))
        tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
        tf_record_writer.write(tf_record.SerializeToString())
    tf_record_writer.close()


def transform(args):
    df, categories = clean_data(args.input)
    scaled, scaler = scale_data(df)
    write(scaled, args.output)
    print(scaled)
    print('done')

    
def parse_args():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument(
        "--input",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str
    )
    return parser.parse_args()

    
def main():
    args = parse_args()
    print("Loaded arguments:")
    print(args)
    transform(args)
    
    
if __name__ == '__main__':
    print("Preprocessing")
    main()
