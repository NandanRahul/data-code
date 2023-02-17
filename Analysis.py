import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('csv1.csv')

data['StartDate'] = pd.to_datetime(data.StartDate)
data['StatusCreatedDate'] = pd.to_datetime(data.StatusCreatedDate)
data['year'] = pd.DatetimeIndex(data.StartDate).year
data['month'] = pd.DatetimeIndex(data.StartDate).month

# computing no of days and time left to start date of each event
data['time_to_event'] = pd.DatetimeIndex(data['StartDate']) - pd.DatetimeIndex(data['StatusCreatedDate'])
data['days_to_event'] = data['time_to_event'] / np.timedelta64(1, 'D')

# dropping rows where ticket was booked after the start date
data = data.drop(data[data['StatusCreatedDate'] >= data['StartDate']].index)

# dropping rows where the customer has booked and is not attending the Event
data_new = data.drop(data[data['BookingStatus'] != 'Attending'].index)


def search_by_event_name(event_name_choice):
    values = data_new['EventName'].unique()
    for i in values:
        if i == event_name_choice:
            # start date of booking and start date of the event is evaluated below
            limiting_date_min = data_new.loc[data_new['EventName'].isin([i]), 'StatusCreatedDate'].min()
            limiting_date_max = data_new.loc[data_new['EventName'].isin([i]), 'StartDate'].max()
            print(limiting_date_max)
            print(limiting_date_min)
            # grouping and summing no of booking according to each month
            data_event_based = data_new[data_new.EventName == i][['GroupSize']]
            grouped = data_event_based.groupby(data_new['StatusCreatedDate'].dt.month)
            results = grouped.sum()
            print(results)
            # trying to plot a bar graph if possible
            # since if data is null then index error will show up
            try:
                results.plot(kind='bar')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                # need to fix position later #todo
                plt.text(-2, -.7, 'Start date of event: ' + str(limiting_date_max) + '\n'
                         + 'Booking commencement date: ' + str(limiting_date_max))
                plt.show()
            except IndexError:
                results.plot(kind='line')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                plt.text(-2, -.7, 'Start date of event: ' + str(limiting_date_max) + '\n'
                         + 'Booking commencement date: ' + str(limiting_date_max))
                plt.show()


def search_by_event_type(event_type_choice):
    values_event_type = data_new['EventType'].unique()
    for i in values_event_type:
        if i == event_type_choice:
            dataframe2021 = data_new[data_new.year == 2021][['EventType', 'month', 'GroupSize']]
            data_event_based = dataframe2021[dataframe2021.EventType == i][['GroupSize']]
            grouped1 = data_event_based.groupby(data['month'])
            dataframe2022 = data_new[data_new.year == 2022][['EventType', 'month', 'GroupSize']]
            data_event_based2 = dataframe2022[dataframe2022.EventType == i][['GroupSize']]
            grouped2 = data_event_based2.groupby(data['month'])
            results1 = grouped1.sum()
            results2 = grouped2.sum()
            print(i)
            print(results1)
            print(results2)
            try:
                results1.plot(kind='bar')
                results2.plot(kind='bar')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                plt.show()
            except IndexError:
                results1.plot(kind='line')
                results2.plot(kind='line')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                plt.show()


# choice 1 means event Name based analysis is given
choice = int(input('please enter your search method: '))
if choice == 1:
    eventNameChoice = input("Please enter the preferred Event Name")
    search_by_event_name(eventNameChoice)

# choice 2 gives event type based analysis showing performance throughout the year.
elif choice == 2:
    eventTypeChoice = input("Please enter your preferred event type")
    search_by_event_type(eventTypeChoice)

