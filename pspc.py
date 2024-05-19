import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import datetime

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # Skip the header row
        for row in csvFileReader:
            date_str = row[0]
            date = datetime.datetime.strptime(date_str, '%m/%d/%Y').date()
            dates.append(date.year)  # Assuming you want to use the year for prediction

            # Clean the price data
            price_str = row[1].strip().replace('$', '').replace(',', '')
            try:
                price = float(price_str)
                prices.append(price)
            except ValueError as e:
                print(f"Error converting {price_str} to float: {e}")

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    x = np.array([x]).reshape(-1, 1)  # Reshape x to be a 2D array

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data("HistoricalQuotes.csv")
predicted_price = predict_prices(dates, prices, 2023)

print(predicted_price)