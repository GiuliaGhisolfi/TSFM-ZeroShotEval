import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from gift_eval.data import Dataset


def load_gift_data(colab: bool = True):
    # make directory for datasets if it does not exist
    if not os.path.exists("data/gift_benchmark"):
        os.makedirs("data/gift_benchmark")
    
    if colab:
        local_dir = "data/gift_benchmark"
        os.makedirs(local_dir, exist_ok=True)

        snapshot_download(
            repo_id="Salesforce/GiftEval",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    # Load environment variables
    load_dotenv()

    # Get the GIFT_EVAL path from environment variables
    if colab:
        gift_eval_path = Path("data/gift_benchmark")
    else:
        gift_eval_path = os.getenv("GIFT_EVAL")

    if gift_eval_path:
        # Convert to Path object for easier manipulation
        gift_eval_path = Path(gift_eval_path)

        # Get all subdirectories (dataset names) in the GIFT_EVAL path
        dataset_names = []
        for dataset_dir in gift_eval_path.iterdir():
            if dataset_dir.name.startswith("."):
                continue
            if dataset_dir.is_dir():
                freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                if freq_dirs:
                    for freq_dir in freq_dirs:
                        dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
                else:
                    dataset_names.append(dataset_dir.name)

        print("Available datasets in GIFT_EVAL:")
        for name in sorted(dataset_names):
            print(f"- {name}")
    else:
        print(
            "GIFT_EVAL path not found in environment variables. Please check your .env file."
        )

def gift_data_to_df(ds_name: str, test_data_only: bool = True) -> pd.DataFrame:
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

    test_split_iter = dataset.test_data
    test_data = []
    for x in test_split_iter:
        x0, x1 = x
        test_data.append(x0)
        test_data.append(x1)
    test_df = pd.DataFrame(test_data)
    test_df["set"] = "test"

    if test_data_only:
        return test_df
    
    else:
        train_split_iter = dataset.training_dataset
        val_split_iter = dataset.validation_dataset
    

        train_data = [x for x in train_split_iter]
        train_df = pd.DataFrame(train_data)
        train_df["set"] = "train"

        val_data = [x for x in val_split_iter]
        val_df = pd.DataFrame(val_data)
        val_df["set"] = "val"

    # concatenate the dataframes
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return df