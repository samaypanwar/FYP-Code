import numpy as np

# Size of the data
train_size = 50_000  # Size of the training set
test_size = 10_000 # Size of the test set

maturities = np.array([1 / 24, 1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 10, 20])
maturities_label = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y', '20Y']

number_of_maturities = len(maturities)
number_of_coupon_rates = 11

coupon_range = np.round(np.linspace(start = 0, stop = 0.1, num = number_of_coupon_rates), 3)

expiries = {
        "1M": "2023-03-07",
        "3M": "2023-05-04",
        '6M': "2023-08-03",
        '1Y': "2024-01-25",
        '2Y': "2025-01-31",
        "3Y": "2026-01-15",
        "5Y": "2028-01-31",
        '10Y': "2032-11-15",
        '20Y': "2042-11-15",
        "30Y": "2052-11-15"
        }

coupons = {
        "1M": 0,
        '3M': 0,
        "6M": 0,
        "1Y": 0,
        '2Y': 4.125 / 100,
        '3Y': 3.875 / 100,
        '5Y': 3.5 / 100,
        '10Y': 4.125 / 100,
        '20Y': 4 / 100,
        '30Y': 4 / 100
        }