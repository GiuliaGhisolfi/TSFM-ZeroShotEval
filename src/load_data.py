import pandas as pd

from gift_eval.data import Dataset


def load_gift_data(ds_name: str):
    """
    Load the GIFT dataset.

    Parameters:
    ds_name (str): The name of the dataset to load.

    Returns:
    pd.DataFrame: A DataFrame containing the dataset.
    """
    # Load the dataset
    to_univariate = False  # Whether to convert the data to univariate
    term = "short"

    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

    train_split_iter = dataset.training_dataset
    val_split_iter = dataset.validation_dataset
    test_split_iter = dataset.test_data

    train_data = [x for x in train_split_iter]
    train_df = pd.DataFrame(train_data)
    train_df["set"] = "train"

    val_data = [x for x in val_split_iter]
    val_df = pd.DataFrame(val_data)
    val_df["set"] = "val"

    test_data = []
    for x in test_split_iter:
        x0, x1 = x
        test_data.append(x0)
        test_data.append(x1)
    test_df = pd.DataFrame(test_data)
    test_df["set"] = "test"

    # concatenate the dataframes
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return df