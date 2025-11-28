from torch.utils.data import Dataset
from datasets import Dataset as DatasetPy
import numpy as np
from enum import Enum


class DatasetColumns(Enum):
    EDGE_INDEX = "edge_index"
    EDGE_ATTR = "edge_attr"
    NODE_FEAT = "node_feat"
    NUM_NODES = "num_nodes"
    INPUT_NODES = "input_nodes"
    ATTN_BIAS = "attn_bias"
    ATTN_EDGE_TYPE = "attn_edge_type"
    SPATIAL_POS = "spatial_pos"
    IN_DEGREE = "in_degree"
    OUT_DEGREE = "out_degree"
    INPUT_EDGES = "input_edges"
    LABELS = "labels"


class InMemoryDataset(Dataset):
    columns = [col.value for col in DatasetColumns]

    def __init__(self, dataset: DatasetPy):
        super().__init__()
        self.data = {}
        for key in self.columns:
            if key not in dataset.column_names:
                raise ValueError(
                    f"Key {key} not found in dataset columns {dataset.column_names}"
                )
        for key in dataset.column_names:
            if key not in self.columns:
                print(f"Warning: Key {key} not in expected columns {self.columns}")

        for key in dataset.column_names:
            print(f"Processing key: {key}")
            val = dataset[key]
            if key == DatasetColumns.EDGE_INDEX.value:
                val = [np.array(v, dtype=np.int32) for v in val]
            elif key == DatasetColumns.EDGE_ATTR.value:
                val = [np.array(v, dtype=np.int32) for v in val]
            elif key == DatasetColumns.NODE_FEAT.value:
                val = [np.array(v, dtype=np.int32) for v in val]
            elif key == DatasetColumns.INPUT_NODES.value:
                val = [np.array(v, dtype=np.int32) for v in val]
            elif key == DatasetColumns.ATTN_BIAS.value:
                val = [np.array(v, dtype=np.float64) for v in val]
            else:
                val = np.array(val, dtype=np.int64)

            self.data[key] = val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.columns}
