from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding as HfDataCollatorWithPadding

from .utils import from_config

class CustomDataset(Dataset):
    """Dataset for CZ4034 - Money, Food and Service Elements Extraction task.

    Parameters
    ----------
    tokenizer
        Tokenizer.
    paths : str or list[str]
        Paths to reviews data files.
    p_augmentation : float
        Probability of performing random replacement on POI/street data.
    cls_token : str
        String representation of "cls" token. Might be "<s>" or "[CLS]", depending on the tokenizer.
    sep_token : str
        String representation of "sep" token. Might be "</s>" or "[SEP]", depending on the tokenizer.
    """
    @from_config(requires_all=True)
    def __init__(self, tokenizer, paths, cls_token="[CLS]", sep_token="[SEP]"):
        super(CustomDataset, self).__init__()

        self.tokenizer = tokenizer
        self._cls_idx = tokenizer.convert_tokens_to_ids(cls_token)[0]
        self._sep_idx = tokenizer.convert_tokens_to_ids(sep_token)[0]

        # Read input
        if isinstance(paths, str):
            paths = [paths]

        dfs = []
        for path in paths:
            df = pd.read_csv(path)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        self.df = self.df.dropna(subset=["Review"])
        self.df = self.df.sample(frac=1)

    def __len__(self):
        return len(self.df)

    def _get_item_for_training(self, idx):
        data_info = self.df.iloc[idx]

        # Base infos
        info = data_info[["Review", "Food", "Service", "Price"]].copy()

        # Tokens
        token_idxs = self.process_input(self.tokenizer, info["Review"])
        food, service, price = info[["Food", "Service", "Price"]]

        return {
            "input_ids": token_idxs, "attention_mask": [1] * len(token_idxs),
            "food_score_label": food, "service_score_label": service, "price_score_label": price
        }

    def __getitem__(self, idx):
        return self._get_item_for_training(idx)

    @staticmethod
    def process_input(tokenizer, text):
        tokens = tokenizer.tokenize(text)
        token_idxs = tokenizer.convert_tokens_to_ids(tokens)[:tokenizer.model_max_length - 2]
        token_idxs = [tokenizer.cls_token_id] + token_idxs + [tokenizer.sep_token_id]
        return token_idxs


class DataCollatorWithPadding(HfDataCollatorWithPadding):
    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Wrap the parent's call function by selectively pass dict of list/tensors to it"""

        selected_features, filtered_features = [], []
        for feature in features:
            selected_feature, filtered_feature = {}, {}
            for key, value in feature.items():
                if isinstance(value, (list, torch.Tensor)):
                    selected_feature[key] = value
                else:
                    filtered_feature[key] = value

            selected_features.append(selected_feature)
            filtered_features.append(filtered_feature)

        # Collate selected features
        selected_features = super(DataCollatorWithPadding, self).__call__(selected_features)

        # Collate filtered features
        all_keys = [tuple(filtered_feature.keys()) for filtered_feature in filtered_features]
        assert len(set(all_keys)) == 1
        all_keys = all_keys[0]
        collated_filtered_features = {}

        for key in all_keys:
            collated_filtered_feature = [filtered_feature[key] for filtered_feature in filtered_features]
            try:
                collated_filtered_feature = torch.tensor(collated_filtered_feature)
            except Exception:
                pass
            collated_filtered_features[key] = collated_filtered_feature

        # Combine everything together
        collated_features = {}
        collated_features.update(selected_features)
        collated_features.update(collated_filtered_features)

        return collated_features
