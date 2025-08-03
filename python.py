import pandas as pd
import numpy as np

# the amount of data to be created
n_samples = 100

# create input data
np.random.seed(42)
building_area = np.random.randint(50, 300, n_samples)
number_of_rooms = np.random.randint(2, 6, n_samples)
age_of_the_house = np.random.randint(1, 30, n_samples)

# for house prices
house_prices = (building_area *5 ) + (number_of_rooms * 20) - (age_of_the_house * 0.5) + np.random.rand(n_samples) * 50
house_prices = house_prices.round(2)

# create a data frame and save it
df = pd.DataFrame({
    'building_area': building_area,
    'numbers_of_rooms': number_of_rooms,
    'age_of_the_house': age_of_the_house,
    'house_price': house_prices
})
df.to_csv('data/home_data.csv', index=False)
print("Dataset berhasil dibuat dan disimpan di data/home_data.csv")