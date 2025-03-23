import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def load_onehot_data(filepath: str) -> pd.DataFrame:
    """
    Load a one-hot encoded CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame loaded from the CSV.
    """
    # Adjust header parameter as needed (header=0 if your CSV has headers)
    return pd.read_csv(filepath, header=0)


def clean_data_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all values in the dataframe to boolean.
    Any non-zero value will be converted to True, and zero to False.

    Parameters:
        df (pd.DataFrame): DataFrame with numeric values.

    Returns:
        pd.DataFrame: DataFrame with boolean values.
    """
    return df.applymap(lambda x: x != 0)


def generate_frequent_itemsets(
    df: pd.DataFrame, min_support: float = 0.10
) -> pd.DataFrame:
    """
    Generate frequent itemsets using the Apriori algorithm.

    Parameters:
        df (pd.DataFrame): One-hot encoded (boolean) DataFrame.
        min_support (float): Minimum support threshold.

    Returns:
        pd.DataFrame: Frequent itemsets.
    """
    return apriori(df, min_support=min_support, use_colnames=True)


def generate_rules_from_itemsets(
    itemsets: pd.DataFrame, min_confidence: float = 0.40
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets with a minimum confidence threshold.

    Parameters:
        itemsets (pd.DataFrame): Frequent itemsets DataFrame.
        min_confidence (float): Minimum confidence threshold.

    Returns:
        pd.DataFrame: Association rules sorted by lift in descending order.
    """
    rules = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    return rules.sort_values(by="lift", ascending=False).reset_index(drop=True)


def get_top_rules(rules: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Return the top N association rules based on lift.

    Parameters:
        rules (pd.DataFrame): DataFrame of association rules.
        top_n (int): Number of top rules to return.

    Returns:
        pd.DataFrame: Top N association rules.
    """
    return rules.head(top_n).reset_index(drop=True)


def find_redundant_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify redundant (compressible) rules among the given association rules.

    A rule is considered redundant if there is another rule with a strict subset
    of its antecedents that predicts the same consequents.

    Parameters:
        rules_df (pd.DataFrame): DataFrame of association rules.

    Returns:
        pd.DataFrame: A DataFrame of redundant rules.
    """
    redundant_indices = []
    for i, rule_i in rules_df.iterrows():
        ant_i, cons_i = set(rule_i["antecedents"]), set(rule_i["consequents"])
        for j, rule_j in rules_df.iterrows():
            if i == j:
                continue
            ant_j, cons_j = set(rule_j["antecedents"]), set(rule_j["consequents"])
            if ant_j < ant_i and cons_i == cons_j:
                redundant_indices.append(i)
                break
    return rules_df.loc[redundant_indices]


def main():
    # Adjust the filepath to your CSV location
    filepath = "data/GroceryPurchases-1.csv"

    # Step 1: Load the one-hot encoded data
    df = load_onehot_data(filepath)

    # Step 1.1: Clean the data by converting nonzero values to True (and zeros to False)
    df_bool = clean_data_to_boolean(df)

    # Step 2: Generate frequent itemsets (min support 10%)
    frequent_itemsets = generate_frequent_itemsets(df_bool, min_support=0.10)

    # Step 3: Generate association rules (min confidence 40%)
    rules = generate_rules_from_itemsets(frequent_itemsets, min_confidence=0.40)

    # Step 4: Select the top 25 rules by lift
    top25_rules = get_top_rules(rules, top_n=25)

    # Step 5: Identify redundant (compressible) rules among the top 25
    redundant_rules = find_redundant_rules(top25_rules)

    # Display results
    print("Top 25 Association Rules (sorted by lift):")
    print(top25_rules)
    print("\nRedundant (compressible) Rules among the Top 25:")
    if redundant_rules.empty:
        print("None found.")
    else:
        print(redundant_rules)


if __name__ == "__main__":
    main()
