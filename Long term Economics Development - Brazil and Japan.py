"""
Created on Fri Mar 10 14:56:17 2023
"""

# Import packages.
import numpy as np # For working with data.
import pandas as pd # For working with data. 
import matplotlib.pyplot as plt # for plotting charts
import statsmodels.api as sm # for HP filter
import seaborn as sns # for slightly prettier charts
from typing import Union # for use in regression
sns.set_theme('talk', style = 'white')

# The following code is the combination of three separate code files, which produce 
# three separate charts: one for economic growth, one for TFP, and one for 
# the growth decomposition exercise. This document will be organised as follows:
    # Section 1: TFP. 
    # Section 2: Real GDP and trendline. 
    # Section 3: Growth decomposition. 
    

# This code creates charts for the following country: 
country = "Japan" # Include name in double quote marks.
# Change Japan for Brazil to make the analysis on Brazil. 

# Section 1: Total factor productivity 

# Part 1 Loading and cleaning the data.

# Loading the data as an Excel file. 
data = pd.read_excel(r'/Users/louisleibovici/Desktop/pwt100.xlsx', sheet_name = "Data", header = (0))

# Cleaning data by subsetting relevant columns and compute real GDP in 2017 prices. 
data = data.loc[:, ["country", "year", "rgdpna", "rnna", "emp", "hc", "labsh", "avh", "pl_c"]] # to index the data

# Specify country and slect relevant variables for later calculations. 
data = data.loc[data["country"] == country, ("country", "year", "rgdpna", "rnna", "emp", "hc", "labsh", "avh", "pl_c")]

# Reset the index. This sets the indices in order, making it easier to work with the dataframe. 
data = data.reset_index(drop = True)

# Subsetting the time series data with an earliest and latest year. 
ymax = 2019
ymin = 1950

# Using logical indexing, we can subset the data to rows of years between ymin and ymax.
Y = data.loc[np.logical_and(data["year"] <= ymax, data["year"] >= ymin), ["country", "year", "rgdpna", "rnna", "emp", "hc", "labsh"]]
data = data[data["year"] >= ymin] # we keep only years after ymin



# Part 2 Constructing the variables.

# First, we define the exponents as each factor's share of income.
alpha = 1 - data["labsh"]  # the capital exponent in the Solow model.
beta = data["labsh"] # the labour exponent in the Solow model. 

# Assigning complete set of variables to describe the Solow growth model. 
country = data["country"][0] # for chart names
Y = data["rgdpna"] 
K = data["rnna"]
L = data["emp"] * data["hc"] * data["avh"] 

# We will standardise our data to yield TFP = 1 at 1950. This is important: not only does
# this give an intuitive understanding to the values of our estimates as being relative to
# the year 1950, it also allows us to weight factors by their shares of income via the exponents. 
# This is needed because otherwise, our estimate of labour as the number of hours worked times their 
# quality will be far higher than capital, and it would be not be expressed in terms of prices, 
# whereas capital is. Our approach avoids this problem.

K_standardised = 100 * K / K[0]

L_standardised = 100 * L / L[0]

Y_standardised = 100 * Y / Y[0]


# Calculating total factor productivity by the Solow model. 
denominator = (K_standardised ** alpha) * (L_standardised ** beta)
tfp = Y_standardised / denominator

# Applying the HP Filter to create a trendline. Our focus only concerns the long run so we 
# use the HP Filter to provide a trendline that removes short-term noise. Besides being used by 
# the IMF and ECB to estimate 'long run' output, this filter appears to do a good job visually. 
# For our purposes, it works well.


tfp_hp = sm.tsa.filters.hpfilter(tfp, lamb=100) # 100 is a popular choice for the 
# smoothing parameter when applying the HP Filter to annual data.
tfp_smoothed = tfp_hp[1] # this specifies the trend data, rather than the cyclical data.

# Setting the theme with Seaborn. 
sns.set(style="darkgrid") # sets a dark grid for our chart. 


# Plotting the data. 

fig, (ax1) = plt.subplots(1, 1, figsize = (5,5), dpi=1000) # The dpi argument sets the resolution

# Plot absolute TFP and its trendline. 
ax1.plot(data['year'], tfp, linewidth = 2, label = 'Estimate')
ax1.plot(data['year'], tfp_smoothed, linewidth = 2, label = 'Trend')
ax1.set_ylabel('Level')
ax1.set_title('TFP for ' + str(country) + ', 1950-2019, 1950=1')
ax1.legend()




# Section 2: Real output and its trendline

# Part 1: Adjusting the time series to 1960-2019.
# We choose these years because our additive quadratic trendline specification does not work well when starting
# from 1950 because it selects an unusually low intercept for Japan. Omitting the 1950s is not crucial here
# as our aim is to depict a useful trendline, and because growth rates are covered by the growth accounting exercise. 

# Reset the index. 
data = data.reset_index(drop = True)

# Subset the time series data to specify the earliest and latest year. 
ymax = 2019
ymin = 1960

# Using logical indexing, we can subset the data to rows of years between ymin and ymax.
Y = data.loc[np.logical_and(data["year"] <= ymax, data["year"] >= ymin), "rgdpna"]
data = data[data["year"] >= ymin] # we only keep years after ymin


# Part 2 Assign variables. 

# Compute log GDP.
y = np.log(Y)

# Define the length of time, which is the sample size for our regression.
T = len(Y) 
T_all = data["year"].max() - (ymin - 1) # number of all years in the data after ymin



# Part 3 Compute a trendline.

# Define a function for obtaining regression coefficients. 

def get_regression_coefs(dependent_variable: np.ndarray, regressor1: np.ndarray, regressor2: np.ndarray, regressor3: Union[np.ndarray, None] = None) -> np.ndarray:
    X = np.concatenate((regressor1[:, None], regressor2[:, None], regressor3[:, None]), axis=1)

    XX = X.T @ X # Construct X'X
    XY = X.T @ dependent_variable # Construct X'Y, 

    coefs = np.linalg.inv(XX) @ XY # solve for formula (X'X)^(-1) X'Y, as in EC2C1. 

    return coefs


# Note that the above function makes use of vector operations, which are computationally more efficient.
# Now we will create a trendline following an additive quadratic model. 

# Use NumPy to initialise regressor arrays. 
x1 = np.ones(T)
x2 = np.arange(1, T+1)
x3 = np.power(x2, 2)

# Obtain the regression coefficients. 
a_add_quad, b1_add_quad, b2_add_quad = get_regression_coefs(Y, x1, x2, x3)

# Obtain predicted values, yhat
Yhat_add_quad = a_add_quad + b1_add_quad * np.arange(1, T_all+1) + b2_add_quad * np.power(np.arange(1, T_all+1), 2)

# Finally, convert into log-units
log_Yhat_add_quad = np.log(Yhat_add_quad)


# Part 4: Plotting the Data
lw = 3 # Setting linewidth to be consistent across all plots. 

# Setting the theme with Seaborn. 
sns.set(style="darkgrid")

# Plotting the data and trendline. DPI argument sets the resolution. 
fig, (ax1) = plt.subplots(1,1, figsize = (5,5), dpi=1000)

ax1.plot(data['year'], np.log(data['rgdpna']), linewidth = lw, label = 'Real GDP')
ax1.plot(data['year'], log_Yhat_add_quad, linewidth = lw, linestyle = 'dashed', label = 'Trendline')
ax1.set_title('Log real GDP for ' + country + ', 2017 prices, 1960 to 2019')
ax1.legend()







# Section 3: Growth decomposition.

# Part 1 Loading and cleaning the data.

# Reset the index.
data = data.reset_index(drop = True)

# Subsetting the time series data with an earliest and latest year. 
ymax = 2019
ymin = 1950

# Using logical indexing, we can subset the data to rows of years between ymin and ymax.
Y = data.loc[np.logical_and(data["year"] <= ymax, data["year"] >= ymin), ["country", "year", "rgdpna", "rnna", "emp", "hc", "labsh"]]
data = data[data["year"] >= ymin] # we keep only years after ymin


# Part 2 Constructing the variables.

# Assigning complete set of variables to describe the Solow growth model. 
country = data["country"][0] # for chart names
Y = data["rgdpna"] # real GDP
K = data["rnna"] # capital stock
L = data["emp"] * data["hc"] * data["avh"] # labour 'stock' = total hours worked times PWT's human capital index
alpha = 1 - data["labsh"]  # the capital exponent in the Solow model.
beta = data["labsh"] # the labour exponent in the Solow model. 
A = Y / ((K ** alpha) * (L ** beta)) # total factor productivity
T = len(Y) # the number of years

# Calculating a smoothed GDP trendline using the HP filter, for less noisey growth accounting. 
rgdp_hp = sm.tsa.filters.hpfilter(data["rgdpna"], lamb=6.25) # 6.25 is smoothing parameter appropriate for annual data
Y_smoothed = Y - rgdp_hp[0] # Y_smoothed only includes trend component of rgdp


# Part 3 Calculating growth rates.

# Compute the growth rates of each variable. Start by creating empty Numpy arrays.
y_growth = np.empty(T-1) # taking difference removes one observation effectively
y_growth_smoothed = np.empty(T-1)
k_growth = np.empty(T-1)
l_growth = np.empty(T-1)
a_growth = np.empty(T-1)

# Assign growth rate values in percentage terms to the Numpy arrays above. 
for t in range(T-1):
    #if t < t-1:
        y_growth[t] = 100 * (Y[t+1] - Y[t]) / Y[t]
        y_growth_smoothed[t] = 100 * (Y_smoothed[t+1] - Y_smoothed[t]) / Y_smoothed[t]
        k_growth[t] = 100 * (K[t+1] - K[t]) / K[t]
        l_growth[t] = 100 * (L[t+1] - L[t]) / L[t]
        

# Calculating contributions in the standard way for the Solow model. 
k_contribution = k_growth * alpha[1:] 
l_contribution = l_growth * beta[1:]
tfp_contribution = y_growth - k_contribution - l_contribution
tfp_contribution_smoothed = y_growth_smoothed - k_contribution - l_contribution

# Setting data for the x-axis as the years.
years = data['year']


# Part 5 Creating a data frame so that Matplotlib can correctly stack bars. 

# Start by creating an array with the three variables we wish to plot as bars. 
data = np.array([k_contribution, l_contribution, tfp_contribution])

# Take negative and positive values apart and cumulate them.
def get_cumulated_array(bar_data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(bar_data))
    d[1:] = cum[:-1]
    return d  

cumulated_data = get_cumulated_array(data, min = 0)
cumulated_data_neg = get_cumulated_array(data, max = 0)

# Re-merge negative and positive data.
row_mask = (data<0)
cumulated_data[row_mask] = cumulated_data_neg[row_mask]
data_stack = cumulated_data

# For convenience, set a colour list. Colours can be abbreviated by their first letter. 
cols = ["g", "y", "b"]


# Part 6 Plot data. 

# Creating a figure with two plots, one row, two columns and a size of 13 by 6 inches.
fig, (ax1) = plt.subplots(1, 1, figsize=(9,5), dpi=1000)

# Start by creating an array with the three variables we wish to plot as bars. 
data = np.array([k_contribution, l_contribution, tfp_contribution_smoothed])

# Take negative and positive values apart and cumulate them.
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d  

cumulated_data = get_cumulated_array(data, min=0)
cumulated_data_neg = get_cumulated_array(data, max=0)

# Re-merge negative and positive data.
row_mask = (data<0)
cumulated_data[row_mask] = cumulated_data_neg[row_mask]
data_stack = cumulated_data

# Setting the theme with Seaborn. 
sns.set(style="darkgrid")

# Plot our data with GDP growth as a line and factor contributions as stackd bars. 

ax1.plot(years[1:], y_growth_smoothed, color=cols[2], linewidth=1.5, label = 'Real GDP growth')
ax1.bar(years[1:], data[0], bottom=data_stack[0], color=cols[0], label = 'Capital contribution')
ax1.bar(years[1:], data[1], bottom=data_stack[1], color=cols[1], label = 'Labour contribution')
ax1.bar(years[1:], data[2], bottom=data_stack[2], color=cols[2], label = 'TFP contribution')

# Specify other chart attributes.
ax1.set_ylabel('Percent')
ax1.set_title('Growth decomposition for ' + str(country) + ', 1950-2019, smoothed')
ax1.legend()
