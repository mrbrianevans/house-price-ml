import pandas as pd
from pandas.api.types import CategoricalDtype

columns = ['Postcode', 'In Use?', 'Latitude', 'Longitude', 'Population', 'Households', 'Altitude',
           'London zone',
           'Index of Multiple Deprivation', 'Quality', 'Distance to station', 'Average Income']
pdf = pd.read_csv('detailed postcodes.csv', usecols=columns)
pdf = pdf[pdf['In Use?'] == 'Yes'].rename(columns={'Postcode': 'postcode'}).set_index(['postcode'])
del pdf['In Use?']
pdf.info()

print(pdf.count())

pdf.to_csv('slim_postcodes.csv')

pdf = pd.read_csv('slim_postcodes.csv')

df = pd.read_csv('pp-2021.csv')
df.columns = ['uid', 'price', 'date', 'postcode', 'property_type', 'new_build', 'duration', 'PAON',
              'SAON', 'street', 'locality', 'town', 'district', 'county', 'ppdCategory', 'status']
df['date'] = pd.to_datetime(df['date'])
property_type = CategoricalDtype(categories=['O', 'D', 'S', 'T', 'F'], ordered=True)
df['property_type'] = df['property_type'].astype(property_type)
df['property_type_cat'] = df['property_type'].cat.codes
df['new_build'] = df['new_build'].astype('category')
df['new_build_cat'] = df['new_build'].cat.codes
df['duration'] = df['duration'].astype('category')
df['duration_cat'] = df['duration'].cat.codes
df['ppdCategory'] = df['ppdCategory'].astype('category')
df['ppdCategory_cat'] = df['ppdCategory'].cat.codes
df['status'] = df['status'].astype('category')
df['status_cat'] = df['status'].cat.codes
county_categories = list(
    df[(df['duration'] == 'F') & (df['new_build'] == 'N') & (df['property_type'] == 'T')].groupby(
        ['county'])[['price']].mean().sort_values(by='price').round(2).reset_index().reset_index()[
        'county'])
county_type = CategoricalDtype(categories=county_categories, ordered=True)
df['county'] = df['county'].astype(county_type)
df['county_cat'] = df['county'].cat.codes
# df = df.set_index('uid')
df = df[df['price'] < 100_000_000]  # clean massive prices

mdf = pd.merge(left=df[
    ['uid', 'price', 'postcode', 'property_type', 'new_build_cat', 'duration_cat',
     'ppdCategory_cat']], right=pdf, on='postcode').set_index('uid').fillna(
    {'London zone': 0}).dropna()

# mdf.fillna({'London zone': 0})
#
# mdf = mdf.dropna()

print(mdf.head())

mdf.info()

mdf.to_csv('detailed_house_sales.csv')
