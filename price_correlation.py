# -*- coding: utf-8 -*-
"""
Cryptocurrency Price Correlation Analysis

This script computes Pearson correlations between the price histories of different cryptocurrencies.
The price history data is read from files in the `crypto_history` folder. The script generates
correlation matrices, heatmaps, and scatterplots to visualize the relationships between cryptocurrencies.

Configuration values (e.g., time range, cryptocurrency lists) are loaded from a `config.txt` file.
The file format is as follows:
    start_time=1380585600
    end_time=1597449600
    first_graph_start=1546300800
    second_graph_start=1577836800
    large_list=ADA,XXRP,XXBT,XMLN,DASH,XETH,LINK,XLTC
    short_list=XXRP,XXBT,XETH,XLTC

Features:
- Extract hourly price averages or returns for cryptocurrencies.
- Compute and visualize correlation matrices and heatmaps.
- Analyze specified cryptocurrency subsets.

Dependencies:
- numpy, pandas, matplotlib, seaborn, scipy

Created on: Sat Nov 28 21:28:53 2020
Author: Boris
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Configuration variables defined in config.txt
CONFIG_FILE = 'config.txt'
config = {}
with open(CONFIG_FILE, 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        config[key] = value

START_TIME = int(config['start_time'])
END_TIME = int(config['end_time'])
FIRST_GRAPH_SLICE = int(config['first_graph_start'])
SECOND_GRAPH_SLICE = int(config['second_graph_start'])
LARGE_LIST = config['large_list'].split(',')
SHORT_LIST = config['short_list'].split(',')


def extract_mean_price_ph(file, all_hours):
    """
    Extracts the mean hourly price of a given cryptocurrency.

    This function processes a cryptocurrency price history file to compute
    either the mean price for each hour within a specified time range.

    :param file: String path to the cryptocurrency price file.
    :param all_hours: List of timestamps representing hourly intervals
                        over the desired time range.
    :return: tuple: Two lists -
            crypto_time (list): Timestamps corresponding to the hours.
            crypto_price (list): Hourly price average (or "NA" if no trades occurred).

    Note:
        - The input file must be tab-delimited with the following columns:
          close price, volume, timestamp (in seconds).
        - Trades occurring within a given hour contribute to the hourly price average.
    """
    crypto_price = []
    crypto_time = []
    j = 0
    increment = True
    time_file = 0
    with open(file, "r") as close_price_file:
        close_price_file.readline()  # skip header
        while time_file < all_hours[-1] + 3600:
            price_per_hour = []
            is_trade = False
            while True:
                if increment:
                    lineContent = close_price_file.readline().split("\t")
                    time_file = float(lineContent[2])
                if time_file < all_hours[j]:
                    increment = True
                elif time_file >= all_hours[j] and time_file <= all_hours[j] + 3600:
                    is_trade = True
                    price_per_hour.append(float(lineContent[0]))
                    increment = True  # Previously j += 1
                elif time_file > all_hours[j] + 3600:
                    increment = False
                    break

            # Define the mean price depending on whether a trade was found
            if is_trade:
                if len(price_per_hour) > 1:
                    # mean_price = np.mean(price_per_hour)
                    return_price = price_per_hour[-1] - price_per_hour[0]
                else:
                    # mean_price = price_per_hour[0]
                    return_price = "NA"  # 1 value is not enough to compute the return price
            else:
                # mean_price = "NA"
                return_price = "NA"  # 0 value = NA

            crypto_time.append(all_hours[j])
            # crypto_price.append(mean_price)
            crypto_price.append(return_price)
            j += 1

    return crypto_time, crypto_price

def from_pair_to_crypto(pair_name):
    """
    Converts a cryptocurrency pair name into a standalone cryptocurrency symbol.

    This function removes the EUR suffix (either "EUR" or "ZEUR") from
    cryptocurrency pair names used in Kraken trading data.

    :param pair_name: Pair name string, e.g., "XXBTZEUR" or "ADAEUR".
    :return: String, cryptocurrency symbol, e.g., "XXBT" or "ADA".

    Note: Special handling is applied for pairs like "XTZEUR".
    """
    if pair_name[-4:] == "ZEUR" and pair_name != "XTZEUR":
        return pair_name.rstrip("ZEUR")
    elif pair_name[-3:] == "EUR":
        return pair_name.rstrip("EUR")

def corrfunc(x, y, **kws):"""
    Computes the Pearson correlation coefficient and annotates it on a plot.

    This function calculates the correlation coefficient (r) and its 
    significance level (p-value). If significant, an asterisk notation 
    (*, **, ***) is added to indicate the level of significance.

    :param x: First data series.
    :param y: Second data series.

    Annotation:
        - r: Correlation coefficient rounded to 2 decimal places.
        - Significance: 
            *   p <= 0.05
            **  p <= 0.01
            *** p <= 0.001
    """
    nas = np.logical_or(np.isnan(x), np.isnan(y))  # find coordinates of NaNs in x and y
    r, p = stats.pearsonr(x[~nas], y[~nas])  # remove coordinate if either in x or in y
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate("r = {:.2f} {}".format(r, p_stars), xy=(0.05, 0.9), xycoords=ax.transAxes)

def annotate_colname(x, **kws):
    """
    Annotates the column name on the diagonal of a seaborn PairGrid.

    :param x: Data for the diagonal plot.
    """
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
              fontweight='bold')

def cor_matrix(df):
    """
    Creates a comprehensive correlation matrix visualization for a given DataFrame.

    This function generates a seaborn PairGrid with the following features:
    - Upper triangle: Scatterplots with regression lines.
    - Diagonal: Histograms with column name annotations.
    - Lower triangle: Correlation coefficients with significance levels
                      and kernel density plots.

    :param df: DataFrame containing cryptocurrency price data.
    :return: The configured PairGrid for further customization or saving.

    Note:
        - NaN values are ignored when computing correlations.
        - This visualization is useful for detecting patterns and relationships
          between multiple variables.
    """
    sns.set(style='white')
    g = sns.PairGrid(df, dropna=True)
    # Use normal regplot as `lowess=True` doesn't provide CIs.
    g.map_upper(sns.regplot, scatter_kws={'s':8})
    g.map_diag(sns.distplot)
    g.map_diag(annotate_colname)
    g.map_lower(corrfunc)
    g.map_lower(sns.kdeplot, cmap='Blues_d', fill=True)  # Issue with kdeplot not solved completely, but annotation solved by putting it before the plot function
    # Remove axis labels, as they're in the diagonals.
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    return g


if __name__ == "__main__":
    # Define the main variables
    all_hours = list(range(START_TIME, END_TIME, 3600))  # All rounded hours from start time to end time
    crypto_price_ph = dict()

    # Process all the cryptocurrency records
    # List all files in the "crypto_history" directory that contain "close_price" in their filenames
    list_files = [file for file in sorted(os.listdir("crypto_history")) if "close_price" in file]

    # Initialize a list to store cryptocurrency symbols and a dictionary for hourly prices
    cryptocurrencies = []
    for crypto_file in list_files:
        # Extract the cryptocurrency symbol from the file name (everything before the first underscore)
        crypto = crypto_file.split("_")[0]
        cryptocurrencies.append(crypto)  # Add the symbol to the list

        # Initialize a placeholder for storing time and price data for the cryptocurrency
        crypto_price_ph[crypto] = [[], []]

        # Extract hourly mean price and corresponding timestamps for the cryptocurrency
        crypto_price_ph[crypto][0], crypto_price_ph[crypto][1] = extract_mean_price_ph("crypto_history/{}".format(crypto_file), all_hours)
    print("Extraction DONE")

    # Generate dictionaries of prices segmented by year
    prices_2019 = dict()      # Prices for the year 2019
    prices_2020 = dict()      # Prices for the year 2020
    prices_20192020 = dict()  # Prices from 2019 through 2020

    for crypto in cryptocurrencies:
        # Initialize indices for slicing the time series data
        i = 0
        j = 0

        # Find the index corresponding to the start of 2019 (e.g. timestamp: 1546300800)
        i = crypto_price_ph[crypto][0].index(FIRST_GRAPH_SLICE)

        # Convert pair symbols (e.g., BTCZEUR) to cryptocurrency names (e.g., BTC)
        name = from_pair_to_crypto(crypto)

        # Store all prices from 2019 onward in the combined dictionary
        prices_20192020[name] = crypto_price_ph[crypto][1][i:]  # 2019-now

        # Find the index corresponding to the start of 2020 (e.g. timestamp: 1577836800)
        j = crypto_price_ph[crypto][0].index(SECOND_GRAPH_SLICE)

        # Store prices from 2020 onward in the 2020 dictionary
        prices_2020[name] = crypto_price_ph[crypto][1][j:]  # 2020-now

        # Only include cryptocurrencies with valid data for 2019 in the 2019 dictionary
        if [val for val in crypto_price_ph[crypto][1][i:j] if val != "NA"]:  # 2019-2020, take only crypto if any data in 2019
            prices_2019[name] = crypto_price_ph[crypto][1][i:j]
    print("Dictionary split per date")

    # Convert price dictionaries to DataFrames for analysis
    df_2019 = pd.DataFrame(prices_2019)  # DataFrame for 2019 prices
    key_list = [i for i in prices_2019.keys()]  # List of cryptocurrency names in the 2019 dataset
    df_2019[key_list] = df_2019[key_list].replace({'NA':np.nan})  # Replace "NA" with NaN for easier numerical processing
    df_2019_sub = df_2019[LARGE_LIST]  # Subset for cryptocurrencies in the large list
    df_2019_sub2 = df_2019[SHORT_LIST]  # Subset for cryptocurrencies in the short list
    # corrfunc(df_2019[key_list[0]], df_2019[key_list[1]])

    df_2020 = pd.DataFrame(prices_2020)
    key_list = [i for i in prices_2020.keys()]
    df_2020[key_list] = df_2020[key_list].replace({'NA':np.nan})
    df_2020_sub = df_2020[LARGE_LIST]
    df_2020_sub2 = df_2020[SHORT_LIST]

    df_20192020 = pd.DataFrame(prices_20192020)
    key_list = [i for i in prices_20192020.keys()]
    df_20192020[key_list] = df_20192020[key_list].replace({'NA':np.nan})
    df_20192020_sub = df_20192020[LARGE_LIST]
    df_20192020_sub2 = df_20192020[SHORT_LIST]
    print("Conversion to dataframe DONE")

    # Format data for clustermaps
    # Compute correlation matrices for each dataset
    full_corr_matrix_19 = df_2019.corr()
    full_corr_matrix_20 = df_2020.corr()
    full_corr_matrix_1920 = df_20192020.corr()


    ##### ----- Plot the data ----- #####

    ##### ----- Start: Clustermaps ----- #####
    # Clustermap data and corresponding filenames
    clustermap_data = [
        (full_corr_matrix_19, "output_plots/correlation_heatmap_2019.png"),
        (full_corr_matrix_20, "output_plots/correlation_heatmap_2020.png"),
        (full_corr_matrix_1920, "output_plots/correlation_heatmap_20192020.png"),
    ]

    # Generate clustermap plots
    for matrix, filename in clustermap_data:
        g = sns.clustermap(matrix, vmin=-1, vmax=1, center=0, linewidths=.5,
                           cbar_kws={"shrink": .5}, xticklabels=True,
                           yticklabels=True, cmap="RdBu_r")
        g.ax_row_dendrogram.remove()
        plt.savefig(filename, dpi=300)
        plt.clf()  # Clear the figure
        plt.cla()
        plt.close()
    ##### ----- End: Clustermaps ----- #####

    ##### ----- Start: Correlation plots ----- #####
    # Correlation matrix data and corresponding filenames
    correlation_data = [
        (df_2019_sub, "output_plots/correlation_2019.png"),
        (df_2020_sub, "output_plots/correlation_2020.png"),
        (df_20192020_sub, "output_plots/correlation_2019-2020.png"),
        (df_2019_sub2, "output_plots/highest_correlation_2019.png"),
        (df_2020_sub2, "output_plots/highest_correlation_2020.png"),
        (df_20192020_sub2, "output_plots/highest_correlation_2019-2020.png"),
    ]

    # Generate correlation matrix plots
    for df, filename in correlation_data:
        cor_matrix(df)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.clf()  # Clear the figure
        plt.cla()
        plt.close()
    ##### ----- End: Correlation plots ----- #####
