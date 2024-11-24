import sys
import zipfile
import pandas as pd


def read_data(path):
    # TODO: switch to S3 bucket
    zip = zipfile.ZipFile(path)
    fp = zip.open('Historical Product Demand.csv') 
    df = pd.read_csv(fp)

    df.dropna(subset=['Date'], inplace=True)
    df.dropna(subset=['Order_Demand'], inplace=True)
    
    df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
    df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
    df['Order_Demand'] = df['Order_Demand'].astype('int64')

    df['Date'] = pd.to_datetime(df['Date']) 
    df.sort_values('Date', inplace=True)
    return df
    

def select_warehouse(df, warehouse_name):
    return df[df['Warehouse'] == 'Whse_A']


def select_daterange(df, start, stop):
    return df[(df['Date'] >= start)  & (df['Date'] < stop)].reset_index()


def frequent_products(df, th=1000):
    frequent_products = df['Product_Code'].value_counts()[
        df['Product_Code'].value_counts() > th].index
    return df[df.Product_Code.isin(frequent_products)]


def on_grain(df):
    return df[['Product_Code', 'Date', 'Order_Demand']]\
        .groupby(['Product_Code', 'Date'])\
        .sum()\
        .reset_index()


def reindex(df):
    pivot = df.pivot(
        index='Date', columns='Product_Code', values='Order_Demand').fillna(0)    
    idx = pd.date_range(
        start=pivot.index.min(), end=pivot.index.max(), freq='D')
    return pivot.reindex(idx).ffill()


def fill(df):
    return df.replace(to_replace=0, method='ffill')


def original_grain(df):
    df = df.stack().reset_index()    
    df['Date'] = df['level_0']
    df['Volume'] = df[0]
    return df


def write(df, out):
    df.to_csv(out)
    

if __name__ == '__main__':
    path     = sys.argv[1]
    whse     = sys.argv[2]
    start    = sys.argv[3]
    stop     = sys.argv[4]
    original = sys.argv[5] == 'original'
    out      = sys.argv[6]
    
    df = read_data(path)
    df = select_warehouse(df, whse)
    df = select_daterange(df, start, stop)
    df = frequent_products(df)
    df = on_grain(df)
    df = reindex(df)
    df = fill(df)
    if original:
        df = original_grain(df)
    write(df, out)
