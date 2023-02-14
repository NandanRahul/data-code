import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('csv1.csv')
data.sort_values('StartDate')
data['StartDate'] = pd.to_datetime(data.StartDate)
data['StatusCreatedDate'] = pd.to_datetime(data.StatusCreatedDate)
data['year'] = pd.DatetimeIndex(data.StartDate).year


def search_by_event_name(event_name_choice):
    values = data['EventName'].unique()
    for i in values:
        start_date = None
        end_date = None
        if i == event_name_choice:
            date = data[data.EventName == i][['StartDate']]
            dates = data[data.EventName == i][['StatusCreatedDate']]
            dates.sort_values('StatusCreatedDate')
            for k in dates:
                start_date = k
            for j in date:
                end_date = j
            print(start_date, end_date)
            data_event_based = data[data.EventName == i][['GroupSize']]
            grouped = data_event_based.groupby(data['StatusCreatedDate'].dt.month)
            results = grouped.sum()
            print(results)
            try:
                results.plot(kind='bar')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                plt.show()
            except IndexError:
                results.plot(kind='line')
                plt.xlabel('Month')
                plt.ylabel('Bookings')
                plt.show()


def search_by_event_type(event_type_choice):
    values_event_type = data['EventType'].unique()
    for i in values_event_type:
        if i == event_type_choice:
            dataframe2021 = data[data.year == 2021][['EventType', 'month', 'GroupSize']]
            data_event_based = dataframe2021[dataframe2021.EventType == i][['month', 'GroupSize']]
            grouped1 = data_event_based.groupby('month')
            dataframe2022 = data[data.year == 2022][['EventType', 'month', 'GroupSize']]
            data_event_based2 = dataframe2022[dataframe2022.EventType == i][['month', 'GroupSize']]
            grouped2 = data_event_based2.groupby('month')
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


data['StartDate'] = pd.to_datetime(data.StartDate)
data['StatusCreatedDate'] = pd.to_datetime(data.StatusCreatedDate)
data['date'] = data['StatusCreatedDate'].dt.date
data['month'] = pd.DatetimeIndex(data.StartDate).month
choice = int(input('please enter your search method: '))
if choice == 1:
    eventNameChoice = input("Please enter the preferred Event Name")
    search_by_event_name(eventNameChoice)
elif choice == 2:
    eventTypeChoice = input("Please enter your preferred event type")
    search_by_event_type(eventTypeChoice)
