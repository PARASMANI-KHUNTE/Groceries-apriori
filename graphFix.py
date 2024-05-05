import pandas as pd
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt

# Load the dataset from CSV file
df = pd.read_csv('groceries.csv', header=None)

# Perform one-hot encoding
basket_sets = pd.get_dummies(df, prefix='', prefix_sep='')

# Run Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

# Sort the frequent itemsets by support value
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Plotting frequent itemsets as a bar graph
plt.figure(figsize=(12, 8))
plt.barh(frequent_itemsets['itemsets'].astype(str), frequent_itemsets['support'], color='skyblue')
plt.xlabel('Support', fontsize=14)
plt.ylabel('Itemsets', fontsize=14)
plt.title('Frequent Itemsets', fontsize=16)
plt.gca().invert_yaxis()  # Invert y-axis to show highest support at the top

# Adjust spacing and padding
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().tick_params(axis='x', pad=8)
plt.gca().tick_params(axis='y', pad=5)
plt.tight_layout()
plt.show()
