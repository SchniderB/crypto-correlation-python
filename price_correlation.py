#!/home/boris/Documents/Trading_bot/.virtualenv/krakenex/bin/python
# -*- coding: utf-8 -*-

"""
This script computes a Pearson correlation between the BTC price and the price of
any other cryptocurrency.

Created on Sat Nov 28 21:28:53 2020

@author: boris
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

def extract_mean_price_ph(file, all_hours):
    """
    Function that extracts the mean price per hour of a given cryptocurrency.
    """
    os.chdir("/home/boris/Documents/Trading_bot/benchmark_data/correlation")
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
    if pair_name[-4:] == "ZEUR" and pair_name != "XTZEUR":
        return pair_name.rstrip("ZEUR")
    elif pair_name[-3:] == "EUR":
        return pair_name.rstrip("EUR")

def corrfunc(x, y, **kws):
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
  ax = plt.gca()
  ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
              fontweight='bold')

def cor_matrix(df):
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

# Define the main variables
first_hour = 1380585600  # 1st of October 2013 at 00:00:00 is the first hour of the count
last_hour = 1597449600  # 15th of August 2020 at 00:00:00 is the last hour of the count
all_hours = list(range(1380585600, 1597449600, 3600))  # All rounded hours from start time to end time
crypto_price_ph = dict()

# Process all the cryptocurrency records
list_files = [file for file in sorted(os.listdir("..")) if "close_price" in file]

cryptocurrencies = []
for crypto_file in list_files:
    crypto = crypto_file.split("_")[0]
    cryptocurrencies.append(crypto)
    crypto_price_ph[crypto] = [[], []]
    crypto_price_ph[crypto][0], crypto_price_ph[crypto][1] = extract_mean_price_ph("../{}".format(crypto_file), all_hours)
print("Extraction DONE")

# Generate dictionaries per date
prices_2019 = dict()
prices_2020 = dict()
prices_20192020 = dict()
for crypto in cryptocurrencies:
    i = 0
    j = 0
    i = crypto_price_ph[crypto][0].index(1546300800)
    name = from_pair_to_crypto(crypto)
    prices_20192020[name] = crypto_price_ph[crypto][1][i:]  # 2019-now
    j = crypto_price_ph[crypto][0].index(1577836800)
    prices_2020[name] = crypto_price_ph[crypto][1][j:]  # 2020-now
    if [val for val in crypto_price_ph[crypto][1][i:j] if val != "NA"]:  # 2019-2020, take only crypto if any data in 2019
        prices_2019[name] = crypto_price_ph[crypto][1][i:j]
print("Dictionary split per date")

# Convert to dataframe
df_2019 = pd.DataFrame(prices_2019)
key_list = [i for i in prices_2019.keys()]
df_2019[key_list] = df_2019[key_list].replace({'NA':np.nan})
df_2019_sub = df_2019[["ADA", "XXRP", "XXBT", "XMLN", "DASH", "XETH", "LINK", "XLTC"]]
df_2019_sub2 = df_2019[["XXRP", "XXBT", "XETH", "XLTC"]]
# corrfunc(df_2019[key_list[0]], df_2019[key_list[1]])

df_2020 = pd.DataFrame(prices_2020)
key_list = [i for i in prices_2020.keys()]
df_2020[key_list] = df_2020[key_list].replace({'NA':np.nan})
df_2020_sub = df_2020[["ADA", "XXRP", "XXBT", "XMLN", "DASH", "XETH", "LINK", "XLTC"]]
df_2020_sub2 = df_2020[["XXRP", "XXBT", "XETH", "XLTC"]]

df_20192020 = pd.DataFrame(prices_20192020)
key_list = [i for i in prices_20192020.keys()]
df_20192020[key_list] = df_20192020[key_list].replace({'NA':np.nan})
df_20192020_sub = df_20192020[["ADA", "XXRP", "XXBT", "XMLN", "DASH", "XETH", "LINK", "XLTC"]]
df_20192020_sub2 = df_20192020[["XXRP", "XXBT", "XETH", "XLTC"]]
print("Conversion to dataframe DONE")

# Plot the data
full_corr_matrix_19 = df_2019.corr()
full_corr_matrix_20 = df_2020.corr()
full_corr_matrix_1920 = df_20192020.corr()
print(full_corr_matrix_19)
exit()

g = sns.clustermap(full_corr_matrix_19, vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=True, yticklabels=True, cmap="RdBu_r")
g.ax_row_dendrogram.remove()
plt.savefig("correlation_heatmap_2019.png", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()
print(full_corr_matrix_19)

g = sns.clustermap(full_corr_matrix_20, vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=True, yticklabels=True, cmap="RdBu_r")
g.ax_row_dendrogram.remove()
plt.savefig("correlation_heatmap_2020.png", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

g = sns.clustermap(full_corr_matrix_1920, vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=True, yticklabels=True, cmap="RdBu_r")
g.ax_row_dendrogram.remove()
plt.savefig("correlation_heatmap_20192020.png", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_2019_sub)
plt.savefig("correlation_2019.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_2020_sub)
plt.savefig("correlation_2020.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_20192020_sub)
plt.savefig("correlation_2019-2020.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_2019_sub2)
plt.savefig("highest_correlation_2019.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_2020_sub2)
plt.savefig("highest_correlation_2020.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()

cor_matrix(df_20192020_sub2)
plt.savefig("highest_correlation_2019-2020.png", bbox_inches="tight", dpi=300)
plt.clf()#clears the figure to be able to write the next plot
plt.cla()
plt.close()
