import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# save csv1 and csv2 locations here
location1 = None
location2 = None


# Define a function to create time series features based on the time series index
def create_features(df_prediction):
    dfs_new = df_prediction.copy()
    # Extract hour, day of week, quarter, month, year, day of year, day of month, and week of year from the index
    dfs_new['hour'] = dfs_new.index.hour
    dfs_new['dayofweek'] = dfs_new.index.dayofweek
    dfs_new['quarter'] = dfs_new.index.quarter
    dfs_new['month'] = dfs_new.index.month
    dfs_new['year'] = dfs_new.index.year
    dfs_new['dayofyear'] = dfs_new.index.dayofyear
    dfs_new['dayofmonth'] = dfs_new.index.day
    dfs_new['weekofyear'] = dfs_new.index.isocalendar().week
    return dfs_new


def event_type(df_2):
    # Load the input data, set the index to 'StatusCreatedDate', and resample by week
    df_prediction = df_2.set_index('StatusCreatedDate')
    df_prediction = df_prediction.resample('w').sum()

    # Create time series features on the resampled dataframe
    dfs = create_features(df_prediction)

    # Split the data into training and testing sets
    train, test = train_test_split(dfs, test_size=.3, shuffle=False)

    # Create separate feature and target dataframes for the training and testing sets
    features = ['EventType', 'days_to_event', 'BookingStatus', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    target = 'GroupSize'
    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    # Initialize the XGBoost regressor with some parameters and fit it to the training data
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000, early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
    reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

    # Plot the feature importances of the trained model
    fi = pd.DataFrame(data=reg.feature_importances_, index=reg.get_booster().feature_names, columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()

    # Generate predictions for the test set and plot the raw data and predictions
    test['prediction'] = reg.predict(x_test)
    dfs = dfs.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    ax = dfs[['GroupSize']].plot(figsize=(15, 5))
    test['prediction'].plot(legend=True)
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    plt.show()

    # Calculate and print the RMSE score on the test set
    score = np.sqrt(mean_squared_error(test['GroupSize'], test['prediction']))
    print(f"RMSE Score on Test set: {score:0.2f}")
    test['GroupSize'].plot(legend=True, figsize=(12, 5))
    test['prediction'].plot(legend=True, figsize=(12, 5))


# Read in two CSV files and concatenate them together
data_csv_1 = pd.read_csv(location1)
data_csv_1['EventType'].fillna('missing')
data_csv_1.drop(data_csv_1[data_csv_1['EventType'] == 'missing'].index)
data_csv_2 = pd.read_csv(location2)
# Fill missing values in the 'EventType' column
data_csv_2['EventType'] = data_csv_2['EventType'].fillna("Graduation Ceremony")
data_concat = pd.concat([data_csv_1, data_csv_2], axis=0)

# Convert 'StartDate' and 'StatusCreatedDate' columns to datetime format
data_concat['StartDate'] = pd.to_datetime(data_concat.StartDate)
data_concat['StatusCreatedDate'] = pd.to_datetime(data_concat.StatusCreatedDate)

# Remove rows where 'StatusCreatedDate' is after 'StartDate'
data_temp = data_concat.drop(data_concat[data_concat['StatusCreatedDate'] >= data_concat['StartDate']].index)

# Remove rows where 'BookingStatus' is not 'Attending'
data_concat_new = data_temp.drop(data_temp[data_temp['BookingStatus'] != 'Attending'].index)

# Write merged data to a new CSV file
data_concat_new.to_csv('merged.csv', index=False)

# Read in the merged data from the CSV file
df = pd.read_csv('merged.csv')

# Make a copy of the dataframe
df2 = df.copy()
df2['EventType'].fillna('z')
uni_eve = df2['EventType'].unique()
# Use label encoding to convert categorical columns to numerical values
unique_cols = ['BookingStatus', 'IsLeadAttendee', 'EventType']
encode = preprocessing.LabelEncoder()
for i in unique_cols:
    df2[i] = encode.fit_transform(df2[i])

# Create a new column for the number of days between 'StatusCreatedDate' and 'StartDate'
df2['time_to_event'] = pd.DatetimeIndex(df['StartDate']) - pd.DatetimeIndex(df['StatusCreatedDate'])
df2['days_to_event'] = df2['time_to_event'] / np.timedelta64(1, 'D')

# Sort the dataframe by 'StatusCreatedDate'
df2 = df2.sort_values(by='StatusCreatedDate')

# Set the index of the dataframe to 'StatusCreatedDate'
df2.set_index('StatusCreatedDate', inplace=True)

# Convert the index to datetime format
df2['StatusCreatedDate'] = pd.to_datetime(df2.index, dayfirst=True)

# Create a new column that shows the cumulative sum of 'GroupSize'
df2['cum_booking'] = df2['GroupSize'].cumsum(axis=0)

eve_list = []
for i in uni_eve:
    eve_list.append(i)
eve_list.pop(2)
eve_list.sort()
for i in range(len(eve_list)):
    print(f"{i} : {eve_list[i]}")
choice = input("Please enter your event type from options")
for i in range(len(eve_list)):
    if eve_list[i] == str(choice):
        choice = i
        print(i)
        break
values = df2['EventType'].unique()
for i in values:
    if i == choice:
        df_2_new = df2.drop(df2[df2['EventType'] != i].index)
        df_2_new.to_csv('new.csv', index=False)
        event_type(df_2_new)

