#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import talib as ta


stock_codes1 = ["000001.SS","399001.SZ", "^HSI","^STI","^N225","^KS11","^IXIC","^GSPC","^DJI","IWM",\
                "^GSPTSE","^BVSP","^STOXX50E","^GDAXI", "^FTSE", "^FCHI", "^IBEX","^AXJO","^AORD","^NZ50"]

def calculate_ma(df, ma_periods=[5,10,20,30,60]):
    for period in ma_periods:
        df[f'{period}-day moving average'] = df['Close'].rolling(period).mean()
    return df

stock_codes = stock_codes1[-2:]

for each in stock_codes:
    data = yf.Ticker(each).history(start="2015-01-01", end="2019-12-31", interval="1d")
    df = data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.rename_axis('Date')
    if len(df) == 0:
        continue

    # MA
    df = calculate_ma(df)

    year = list(set(df.index.year))

    # Calculate MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    print(each, df, sep='\n')

    for i in year:
        temp_y = df[df.index.year == i]
        week = list(set(temp_y.index.week))

        for j in week:
            temp_w = temp_y[temp_y.index.week == j]
            has_nan = temp_w.isna().any().any().any()
            if has_nan == 1:
                continue

            # MA chart
            ma_periods = [5, 10, 20, 30, 60]
            ma_plots = [mpf.make_addplot(temp_w[f'{period}-day moving average'], panel=0, ylabel='', title='') for period in ma_periods]

            min_date = temp_w.index.min()
            min_date_tz = min_date.tzinfo
            start_date = pd.to_datetime("20150101").tz_localize(min_date_tz)

            if min_date > start_date:
                # MACD chart
                ap = []
                ap.append(mpf.make_addplot(temp_w['macd'], panel=2, color='fuchsia'))
                ap.append(mpf.make_addplot(temp_w['macd_signal'], panel=2, color='b'))
                colors = ['g' if v <= 0 else 'r' for v in temp_w["macd_hist"]]
                ap.append(mpf.make_addplot(temp_w['macd_hist'], panel=2, type='bar', color=colors))
                s = mpf.make_mpf_style(base_mpf_style='yahoo', mavcolors=['c', 'lime'])
                save_path = r'D:\硕士\project\Progress 3\Data by weeks/'
                filename = f"chart_{each}_{i}_{j}.png"
                mpf.plot(temp_w, addplot=ap+ma_plots, figratio=(40,20), panel_ratios=[2,2,2], type='candle', style=s, volume=True, volume_panel=1, savefig=save_path + filename,
                         ylabel='', ylabel_lower='', title='')


# In[3]:


import numpy as np
import os
from PIL import Image

image_dir = r"D:\硕士\project\Progress 3\Data by weeks"
output_dir = r"D:\硕士\project\Progress 3\good_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = os.listdir(image_dir)
image_list = [i for i in file_list if i.endswith('.png')]
image_list.sort()

for image_file in image_list:
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(output_dir, image_file)
    
    img = Image.open(image_path)
    cropped_img = img.crop((208, 71, 1037, 474))
    cropped_img.save(output_path)

print("Image cropping complete!")


# In[4]:


import pandas as pd
import os
import calendar
import yfinance as yf
from datetime import datetime, timedelta

image_dir = r"D:\硕士\project\Progress 3\good_data"
output_csv = r"D:\硕士\project\Progress 3\dataset\labels.csv"

# Create an empty DataFrame to store the labels
labels_df = pd.DataFrame(columns=["Stock Code", "Year", "Week"])

# Iterate through the images in the directory
image_list = os.listdir(image_dir)
for image_file in image_list:
    if image_file.endswith('.png'):
        # Extract stock code, year, and week from the image filename
        stock_code, year, week = image_file.split("_")[1:4]
        week = week.replace(".png", "")
        
        # Format the week string with leading zero if necessary
        week_str = week if len(week) == 2 else f"0{week}"
        
        # Calculate the start and end dates based on the year and week number
        start_date = datetime.strptime(f"{year}-W{week_str}-1", "%Y-W%W-%w")
        end_date = start_date + timedelta(days=6)
        
        # Load the stock price data from Yahoo Finance
        data = yf.Ticker(stock_code).history(start=start_date, end=end_date)
        if data.empty:
            continue  # Skip to the next image if there is no data
        
        df = data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.rename_axis('Date')
        stock_data = df
                
        # Calculate the label based on the price change
        first_price = stock_data.iloc[0]["Close"]
        last_price = stock_data.iloc[-1]["Close"]
        label1 = 1 if last_price > first_price else 0
        label2 = (last_price-first_price)/first_price
        
        # Append the label to the DataFrame
        new_row = pd.DataFrame([{"Stock Code": stock_code, "Year": year, "Week": week, "Rise": label1, "Return": label2}])
        labels_df = pd.concat([labels_df, new_row], ignore_index=True)

# Save the labels to a CSV file
labels_df.to_csv(output_csv, index=False)

print("Label generation complete!")


# In[5]:


# unzip the file
import os
from zipfile import ZipFile

extracted_dir = "./"
with ZipFile(r"D:/硕士/project/Progress 3/good_data.zip", 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
    

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_dir = r"D:\硕士\project\Progress 3\good_data"
csv_file = "D:\硕士\project\Progress 3\dataset\labels.csv"

# Load the CSV data
data_df = pd.read_csv(csv_file)

# Check the number of missing values in each column
data_df.isnull().sum()

# Create empty lists to store the image data and labels
image_data = []
labels = []

# Iterate through the CSV rows
for index, row in data_df.iterrows():
    stock_code = row["Stock Code"]
    year = row["Year"]
    week = row["Week"]

    # Construct the image filename
    image_filename = f"chart_{stock_code}_{year}_{week}.png"
    image_path = os.path.join(image_dir, image_filename)

    # Open and convert the image to binary format
    image = Image.open(image_path)
    image = image.resize((207, 101))
    image = image.convert('L')
    image_array = np.array(image)

    # Append the image data and label to the lists
    image_data.append(image_array)
    labels.append(row["Rise"])

# Convert the lists to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Split the data into training and testing sets (80:20 ratio)
split_ratio = 0.8
split_index = int(len(image_data) * split_ratio)

image_train_val_data = image_data[:split_index]
label_train_val_data = labels[:split_index]
image_test_data = image_data[split_index:]
label_test_data = labels[split_index:]

# Print the shape of the training and testing sets
print("Training Images:", image_train_val_data.shape)
print("Training Labels:", label_train_val_data.shape)
print("Testing Images:", image_test_data.shape)
print("Testing Labels:", label_test_data.shape)

import copy
data_df_temp = copy.deepcopy(data_df)

