import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 3420  # to get the same test/train split every time


def get_unscaled_dataframe():
    return pd.read_csv('data/detailed_house_sales.csv')


def get_scaled_Xy():
    data = get_unscaled_dataframe()
    x = data[
        ['new_build_cat', 'duration_cat', 'ppdCategory_cat', 'Latitude', 'Longitude', 'Population',
         'Households',
         'Altitude', 'London zone', 'Index of Multiple Deprivation', 'Quality',
         'Distance to station', 'Average Income', 'property_type_D',
         'property_type_F', 'property_type_O', 'property_type_S',
         'property_type_T']].to_numpy()
    y = data['price'].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    return X, y


def get_fixed_test_train_split():
    x, y = get_scaled_Xy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)
    return x_train, x_test, y_train, y_test
