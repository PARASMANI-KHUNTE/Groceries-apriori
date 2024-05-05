from flask import Flask, render_template
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

app = Flask(__name__)

@app.route('/')
def display_association_rules():
    # Load the dataset
    df = pd.read_csv('groceries.csv', header=None)

    # Perform one-hot encoding
    basket_sets = pd.get_dummies(df, prefix='', prefix_sep='')

    # Run Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Convert rules DataFrame to list of dictionaries
    rule_list = rules.to_dict(orient='records')

    return render_template('index.html', rules=rule_list)

if __name__ == '__main__':
    app.run(debug=True)
