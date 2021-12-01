import seaborn as sns
from matplotlib import pyplot as plt

import get_data

pdf = get_data.get_unscaled_dataframe()

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(pdf.corr(), vmax=.8, vmin=-.8, square=True, center=0,
            cmap=sns.diverging_palette(204, 204, as_cmap=True), ax=ax)
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.show()

print('Correlation coefficients of each feature to the price')
print(pdf.corr()['price'].abs().sort_values(ascending=False)[1:])

fig2, ax2 = plt.subplots(figsize=(8, 8))
sns.histplot(data=pdf[(pdf['price'] > pow(10, 4)) & (pdf['price'] < pow(10, 6))], x='price', ax=ax2,
             binwidth=25000)
ax2.ticklabel_format(useOffset=False, style='plain')
ax2.set_title('Distribution of house prices in data set')
plt.savefig('images/distribution_of_prices.png')
plt.show()

print(pdf['price'].describe())
