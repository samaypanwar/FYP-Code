import numpy as np

# Size of the data
train_size = 50_000  # Size of the training set
test_size = 10_000  # Size of the test set

maturities = np.array([1 / 24, 1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 10, 20])
maturities_label = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y', '20Y']

number_of_maturities = len(maturities)
number_of_coupon_rates = 11

coupon_range = np.round(np.linspace(start = 0, stop = 0.1, num = number_of_coupon_rates), 3)
