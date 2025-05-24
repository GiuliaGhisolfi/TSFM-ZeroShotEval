# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator

import datasets
import pyarrow.compute as pc
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ListDataset, ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

TEST_SPLIT = 0.1 # Fraction of the series length to use for testing
MAX_WINDOW = 20 # Maximum number of windows to use for training

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "ME": 18,
    "MS": 18,
    "W": 13,
    "D": 14,
    "H": 48,
    "h": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8, # 1 week
    "D": 30,
    "H": 48,
    "T": 48, # 1 min
    "min": 48,
    "S": 60, # 1 sec
    "s": 60,
    "h": 48,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "h": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
    "min": 8,
}


class Term(Enum):
    """
    Term of the dataset, used to determine the prediction length multiplier.
    - SHORT: 1x prediction length
    - MEDIUM: 10x prediction length
    - LONG: 15x prediction
    """

    SHORT = 1
    MEDIUM = 10
    LONG = 15


load_dotenv()

def get_dataset_path(dataset_name: str) -> Path:
    # Construct the path relative to the script's directory or a base data directory
    # Assumes data is stored in a 'data' directory at the project root
    # You might need to adjust this path based on your project structure
    base_data_dir = Path(__file__).parent.parent.parent / "data"
    return base_data_dir / f"{dataset_name}.hf"


class Dataset:
    """
    Represents a time series dataset loaded from a Hugging Face datasets compatible
    format, prepared for use with GluonTS models.
    """

    def __init__(
        self,
        name: str,
        term: str = Term.SHORT.name,
        to_univariate: bool = True,
    ):
        self.name = name
        self.term = Term[term.upper()]
        self.to_univariate = self._to_univariate if to_univariate else lambda x: x
        self._dataset_path = get_dataset_path(self.name)
        self._hf_dataset = datasets.load_from_disk(str(self._dataset_path))
        self.freq = norm_freq_str(self._hf_dataset.info.features["freq"].to_string())
        self.prediction_length = self._get_prediction_length()
        self._split_dataset()
        self.past_feat_dynamic_real_dim = self._get_past_feat_dynamic_real_dim()

    def _get_prediction_length(self) -> int:
        freq = self.freq
        pred_length = 0

        # Try TFB map first
        if freq in TFB_PRED_LENGTH_MAP:
            pred_length = TFB_PRED_LENGTH_MAP[freq]
        # Then M4 map
        elif freq in M4_PRED_LENGTH_MAP:
            pred_length = M4_PRED_LENGTH_MAP[freq]
        # Then general map
        elif freq in PRED_LENGTH_MAP:
            pred_length = PRED_LENGTH_MAP[freq]
        else:
            # Default or heuristic if frequency is not mapped
            offset = to_offset(freq)
            if offset.n is not None:
                # For frequencies like '10T', '5T' etc.,
                # prediction length is often related to a common period like an hour or day
                # This is a heuristic, adjust if needed based on dataset
                 if offset.n < 60 and ('T' in freq or 'min' in freq or 'S' in freq or 's' in freq): # sub-hourly
                      pred_length = int(60 / offset.n) * 24 # Approx number of steps in a day
                 elif offset.n >= 60 and ('H' in freq or 'h' in freq): # hourly or multi-hourly
                      pred_length = 24 * int(offset.n / 60) # Approx number of hours in a day
                 elif 'D' in freq: # daily
                      pred_length = 7 # Approx number of days in a week
                 elif 'W' in freq: # weekly
                      pred_length = 4 # Approx number of weeks in a month
                 elif 'M' in freq: # monthly
                      pred_length = 12 # Approx number of months in a year
                 else:
                    # Fallback to a small default or raise error
                    print(f"Warning: Could not determine prediction length for frequency {freq}. Using default 30.")
                    pred_length = 30 # Default fallback


            else:
                # Fallback for unhandled complex frequencies
                print(f"Warning: Could not determine prediction length for complex frequency {freq}. Using default 30.")
                pred_length = 30 # Default fallback

        return pred_length * self.term.value


    def _get_past_feat_dynamic_real_dim(self):
         # Placeholder: Implement logic to get the dimension of past_feat_dynamic_real
         # by inspecting a sample entry from the dataset if available.
         # If not available or always None, return 0.
         # For now, return 0 as a default.
         # You might need to load one entry and check its shape.
        if "past_feat_dynamic_real" in self._hf_dataset.column_names:
            # Try to get the dimension from the first non-None entry
            for i in range(len(self._hf_dataset)):
                entry = self._hf_dataset[i]
                if entry.get("past_feat_dynamic_real") is not None:
                    # Assuming past_feat_dynamic_real is a list of lists or similar
                    # Get the dimension of the inner list/array
                    if isinstance(entry["past_feat_dynamic_real"], list) and len(entry["past_feat_dynamic_real"]) > 0:
                         # Assuming inner elements are arrays or lists
                         if isinstance(entry["past_feat_dynamic_real"][0], (list, tuple)):
                              return len(entry["past_feat_dynamic_real"][0])
                         # Assuming it's a list of numerical values for one feature
                         return 1 # Or maybe the length if it's a list of values for one feature?
                                  # This needs clarification based on data format.
                    elif hasattr(entry["past_feat_dynamic_real"], 'shape') and len(entry["past_feat_dynamic_real"].shape) > 1:
                         # Assuming numpy array or similar with shape (time, features)
                         return entry["past_feat_dynamic_real"].shape[1]
                    elif hasattr(entry["past_feat_dynamic_real"], 'shape') and len(entry["past_feat_dynamic_real"].shape) == 1:
                        # Assuming numpy array or similar with shape (time,) for a single feature
                        return 1

        return 0 # Default if column not present or all entries are None/empty

    def _split_dataset(self):
        train_list = []
        test_list = []

        for i, entry in enumerate(self._hf_dataset):
            # Ensure data types are suitable for GluonTS
            # Convert HuggingFace dataset entry (Arrow objects) to Python/NumPy types
            processed_entry = {
                "start": entry["start"].to_pydatetime(),
                "target": entry["target"].to_numpy(),
                "feat_static_cat": entry["feat_static_cat"].to_list() if entry["feat_static_cat"] is not None else None,
                "feat_static_real": entry["feat_static_real"].to_list() if entry["feat_static_real"] is not None else None,
                 # Convert dynamic features to numpy arrays if they exist
                "feat_dynamic_real": entry["feat_dynamic_real"].to_numpy() if entry["feat_dynamic_real"] is not None else None,
                "past_feat_dynamic_real": entry["past_feat_dynamic_real"].to_numpy() if entry["past_feat_dynamic_real"] is not None else None,
            }

            # Determine the split point. Ensure it's an integer index.
            split_idx = max(0, len(processed_entry["target"]) - self.prediction_length)

            # Create train and test entries
            train_entry = processed_entry.copy()
            train_entry["target"] = processed_entry["target"][:split_idx]
             # Ensure dynamic features are also split
            if train_entry["feat_dynamic_real"] is not None:
                train_entry["feat_dynamic_real"] = train_entry["feat_dynamic_real"][:split_idx, :] if train_entry["feat_dynamic_real"].ndim > 1 else train_entry["feat_dynamic_real"][:split_idx]
            if train_entry["past_feat_dynamic_real"] is not None:
                 train_entry["past_feat_dynamic_real"] = train_entry["past_feat_dynamic_real"][:split_idx, :] if train_entry["past_feat_dynamic_real"].ndim > 1 else train_entry["past_feat_dynamic_real"][:split_idx]


            test_entry = processed_entry.copy()
            # The test entry contains the full series including the future part for evaluation
             # Dynamic features are not split here for the test entry as per GluonTS eval format
             # They should cover the full time range of the target

            train_list.append(train_entry)
            test_list.append(test_entry)


        # Wrap the lists in GluonTS ListDataset
        self._train_data = ListDataset(train_list, freq=self.freq)
        # TestData is used here to facilitate evaluation by aligning forecasts with true values
        self._test_data = ListDataset(test_list, freq=self.freq)


    def _to_univariate(self, data_entry: DataEntry) -> DataEntry:
        """Converts a multivariate DataEntry to multiple univariate ones."""
        target = data_entry["target"]
        start = data_entry["start"]
        feat_static_cat = data_entry.get("feat_static_cat")
        feat_static_real = data_entry.get("feat_static_real")
        feat_dynamic_real = data_entry.get("feat_dynamic_real")
        past_feat_dynamic_real = data_entry.get("past_feat_dynamic_real")


        if target.ndim == 1:
            # Already univariate
            yield data_entry
            return

        # Multivariate, split into univariate series
        for i in range(target.shape[0]): # Assuming target shape is (dimensions, time_steps)
            new_entry: DataEntry = {
                "start": start,
                "target": target[i, :],
                 # Static features are the same for all dimensions
                "feat_static_cat": feat_static_cat,
                "feat_static_real": feat_static_real,
                 # Dynamic features need careful handling - if per-dimension, split, otherwise repeat
                 # Assuming feat_dynamic_real and past_feat_dynamic_real are (dimensions, time_steps, features) or (dimensions, time_steps)
                 # This assumption might need adjustment based on the actual dataset format
                "feat_dynamic_real": feat_dynamic_real[i, :, :] if feat_dynamic_real is not None and feat_dynamic_real.ndim > 2 else (feat_dynamic_real[i, :] if feat_dynamic_real is not None and feat_dynamic_real.ndim == 2 else feat_dynamic_real),
                "past_feat_dynamic_real": past_feat_dynamic_real[i, :, :] if past_feat_dynamic_real is not None and past_feat_dynamic_real.ndim > 2 else (past_feat_dynamic_real[i, :] if past_feat_dynamic_real is not None and past_feat_dynamic_real.ndim == 2 else past_feat_dynamic_real),
            }
            # Add original dimension index as a static categorical feature if not already present
            if feat_static_cat is None:
                 new_entry["feat_static_cat"] = [i]
            else:
                 # Append dimension index if feat_static_cat already exists
                 new_entry["feat_static_cat"] = feat_static_cat + [i]


            yield new_entry


    @property
    def train_data(self) -> Iterable[DataEntry]:
        """Iterable over the training data."""
        # Apply univariate transformation if required
        if self.to_univariate.__code__ != (lambda x: x).__code__: # Check if _to_univariate is the actual method
             # Apply the univariate transformation to each entry in the training data
             # Note: This might require iterating and collecting or using a GluonTS transformation chain
             # For simplicity, let's assume _to_univariate can work on an iterable or is applied during split
             # Looking at _split_dataset, the split happens first, then self.to_univariate is applied later... this seems complex.
             # Let's re-evaluate the structure. _split_dataset populates _train_data and _test_data with ListDatasets of dictionaries.
             # The self.to_univariate is then intended to be used *on* these datasets.

             # The correct way to apply transformations in GluonTS is often by wrapping the dataset
             # Let's adjust the property to apply the transformation when accessed
             return Map(self.to_univariate)(self._train_data) # Apply _to_univariate to each item

        return self._train_data # Return original ListDataset if no univariate transform


    @property
    def test_data(self) -> Iterable[DataEntry]:
        """Iterable over the test data."""
        # Apply univariate transformation if required
        if self.to_univariate.__code__ != (lambda x: x).__code__: # Check if _to_univariate is the actual method
             # Apply the univariate transformation to each entry in the test data
             return Map(self.to_univariate)(self._test_data) # Apply _to_univariate to each item

        return self._test_data # Return original ListDataset if no univariate transform


    @property
    def target_dim(self) -> int:
        """Returns the dimensionality of the target time series."""
        # After splitting and potentially converting to univariate,
        # the target dim should be 1 if to_univariate is True, otherwise
        # it's the original dimension from the dataset.
        # We need to get the target dim from the dataset info or a sample.
        # Let's try to get it from the first entry of the HF dataset before any transformations.
        if len(self._hf_dataset) > 0:
            sample_target = self._hf_dataset[0]["target"].to_numpy()
            original_dim = sample_target.shape[0] if sample_target.ndim > 1 else 1
            return 1 if self.to_univariate.__code__ != (lambda x: x).__code__ and original_dim > 1 else original_dim
        return 1 # Default if dataset is empty? Should not happen


# Helper function to process the dataset
def process_dataset(dataset_name: str, term: str, to_univariate: bool) -> Dataset:
    """
    Loads and processes a dataset by name and term.
    """
    return Dataset(name=dataset_name, term=term, to_univariate=to_univariate)

def load_gift_data():
    """
    Helper function to load a specific dataset (example usage, replace with your logic).
    This function might not be needed for the evaluation loop structure
    if you directly instantiate Dataset objects.
    """
    print("Loading GIFT data...")
    # Example usage:
    # dataset = process_dataset("solar/10T", "short", to_univariate=False)
    # print(f"Loaded dataset: {dataset.name}, Freq: {dataset.freq}, Pred Length: {dataset.prediction_length}, Target Dim: {dataset.target_dim}")
    # This function seems just a placeholder in the original notebook based on comments.
    pass # This function doesn't need to do anything for the loop to work


if __name__ == "__main__":
    # Example of how to use the Dataset class
    # (This block runs only if you execute the script directly, not when imported)
    load_dotenv() # Load env variables if running standalone

    try:
        # Example with a known dataset
        # Ensure 'data/solar/10T.hf' exists or adjust path
        dataset = Dataset("solar/10T", term="short", to_univariate=False)
        print(f"Dataset Name: {dataset.name}")
        print(f"Frequency: {dataset.freq}")
        print(f"Prediction Length: {dataset.prediction_length}")
        print(f"Target Dimension: {dataset.target_dim}")
        print(f"Past Feat Dynamic Real Dim: {dataset.past_feat_dynamic_real_dim}")

        print("\nIterating over training data (first 2 entries):")
        for i, entry in enumerate(dataset.train_data):
            print(f"  Train Entry {i+1}: Type={type(entry)}, Keys={list(entry.keys())}, Target Shape={entry['target'].shape}")
            if i >= 1: break

        print("\nIterating over test data (first 2 entries):")
        for i, entry in enumerate(dataset.test_data):
             # This is where the error was happening in the evaluation loop
             print(f"  Test Entry {i+1}: Type={type(entry)}, Keys={list(entry.keys())}, Target Shape={entry['target'].shape}")
             if i >= 1: break


        # Example with univariate conversion
        dataset_uni = Dataset("solar/10T", term="short", to_univariate=True)
        print(f"\nDataset Name (Univariate): {dataset_uni.name}")
        print(f"Target Dimension (Univariate): {dataset_uni.target_dim}")

        print("\nIterating over univariate training data (first 5 entries):")
        for i, entry in enumerate(dataset_uni.train_data):
            print(f"  Train Univariate Entry {i+1}: Type={type(entry)}, Keys={list(entry.keys())}, Target Shape={entry['target'].shape}")
            if i >= 4: break


    except Exception as e:
        print(f"An error occurred during example usage: {e}")