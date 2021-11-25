import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

pdf = pd.read_csv('detailed_house_sales.csv')

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(pdf.corr(), vmax=.8, vmin=-.8, square=True, center=0,
            cmap=sns.diverging_palette(130, 130, as_cmap=True), ax=ax)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')


print('Correlation coefficients of each feature to the price')
print(pdf.corr()['price'].abs().sort_values(ascending=False)[1:])
