import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset from CSV file
df = pd.read_csv('groceries.csv', header=None)

# Perform one-hot encoding
basket_sets = pd.get_dummies(df, prefix='', prefix_sep='')

# Run Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display association rules in a more understandable format
for i, row in rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    
    print(f"Rule: If you buy {antecedents}, then you are likely to buy {consequents}")
    print(f"- Support: {support:.2f}")
    print(f"- Confidence: {confidence:.2f}")
    print(f"- Lift: {lift:.2f}")
    print("="*50)
