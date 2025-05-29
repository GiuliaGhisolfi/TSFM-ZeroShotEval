import json

import numpy as np
import pandas as pd
from gluonts.dataset.common import FieldName, ListDataset
from gluonts.dataset.split import OffsetSplitter, TestData

TERM_MAP = {
    "short": 1,
    "medium": 10,
    "long": 15,
}

def load_data(dataset_name, term="short"):
    with open("data/chronos_dataset_properties.json") as f:
        dataset_properties_map = json.load(f)

    df = pd.read_parquet(f"data/chronos_benchmark/{dataset_name}.arrow")

    prediction_length = dataset_properties_map[dataset_name]["prediction_length"] * TERM_MAP[term]
    frequency = dataset_properties_map[dataset_name]["freq"]
    domain = dataset_properties_map[dataset_name]["domain"]
    num_variates = dataset_properties_map[dataset_name]["num_variates"]

    dataset_gluonts = ListDataset(
        [
            {
                FieldName.ITEM_ID: index,
                FieldName.START: pd.Timestamp(row['start']),
                FieldName.TARGET: np.asarray(row['target'], dtype=np.float32)
            }
            for index, row in df.iterrows()
        ],
        freq=frequency
    )

    dataset = TestData(
        dataset=dataset_gluonts,
        splitter=OffsetSplitter(offset=-prediction_length),
        prediction_length=prediction_length,
        windows=1,
        distance=None,
        max_history=None
    )

    return dataset, prediction_length, frequency, domain, num_variates