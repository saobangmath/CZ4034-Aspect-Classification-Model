import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from .utils import from_config


model_classes = {}


def register_model(cls):
    model_classes[cls.__name__] = cls
    return cls


@register_model
class BertForReviewAspectClassification(nn.Module):
    """Bert model for CZ4034 - Food, money and service extraction task
    Model structure:
                                        |--- foodScore
                         |---- Food ----|
                         |              |--- existence
                         |
    feature_extractor ---|              |--- moneyScore
                         |---- Money ---|
                         |              |--- existence
                         |
                         |              |--- serviceScore
                         |--- Service --|
                         |              |--- existence
    Parameters
    ----------
    model_name_or_path : str
        Path to pretrained model or model identifier from huggingface.co/models.
    config_name : str
        Pretrained config name or path if not the same as model_name.
    tokenizer_name : str
        Pretrained tokenizer name or path if not the same as model_name.
    cache_dir : str
        Path to directory to store the pretrained models downloaded from huggingface.co.
    from_pretrained : bool
        Whether intializing model from pretrained model (other than the pretrained model from huggingface). If yes,
        avoid loading pretrained model from huggingface to save time.
    freeze_base_model : bool
        Whether to freeze the base BERT model.
    fusion : str
        One of ["max_pooling", "average_pooling", "sum"]. How the hidden states from each timestep will be fused
        together to produce a single vector used for binary classifiers (for exist/non-exist of POI/street).
        According to http://arxiv.org/abs/1909.07755, max pooling works best.
    lambdas : list[float]
        Loss weights. Final loss will be computed as: `lambda[0] * poi_span_loss + lambda[1] * poi_existence +
        lambda[2] * street_span_loss + lambda[3] * street_existence `
    """
    @from_config(main_args="model", requires_all=True)
    def __init__(self, model_name_or_path, config_name=None, tokenizer_name=None, cache_dir=None,
                 from_pretrained=False, freeze_base_model=False, fusion="max_pooling", lambdas=[1, 1, 1, 1, 1, 1],
                 mean=0.5, std=3):
        super(BertForReviewAspectClassification, self).__init__()
        # Initialize config, tokenizer and model (feature extractor)
        self.base_model_config = AutoConfig.from_pretrained(
            config_name if config_name is not None else model_name_or_path,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
        )
        if from_pretrained:
            self.base_model = AutoModel.from_config(config=self.base_model_config)
        else:
            self.base_model = AutoModel.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=self.base_model_config,
                cache_dir=cache_dir,
            )

        # Additional layers
        self.mean = mean
        self.std = std
        self._initialize_layers()

        # Fusion
        if fusion not in ["max_pooling", "average_pooling", "sum"]:
            raise ValueError(f"Invalid fusion value. Expected one of ['max_pooling', 'average_pooling', 'sum'], got "
                             f"'{fusion}' instead.")
        self.fusion = fusion

        assert len(lambdas) == 6
        self.lambdas = lambdas

        # Freeze
        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False

    def _initialize_layers(self):
        self.food_score = nn.Linear(self.base_model_config.hidden_size, 1)
        self.food_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.service_score = nn.Linear(self.base_model_config.hidden_size, 1)
        self.service_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.price_score = nn.Linear(self.base_model_config.hidden_size, 1)
        self.price_existence = nn.Linear(self.base_model_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def fusion_layer(self, inp, mask, dim):
        """Fuse model predictions across the sequence length dimension"""
        # max pooling and sum can be handled easily
        if self.fusion in ["max_pooling", "sum"]:
            func = torch.max if self.fusion == "max_pooling" else torch.sum
            epsilon = torch.tensor(1e-16).to(inp)
            inp = torch.where(mask.unsqueeze(-1), inp, epsilon)
            inp = func(inp, dim=dim)
            if not isinstance(inp, torch.Tensor):
                inp = inp[0]
        # average pooling
        elif self.fusion == "average_pooling":
            assert inp.shape == mask.shape
            new_inp = []
            for inp_i, mask_i in zip(inp, mask):
                new_inp.append(inp_i[mask_i].mean())
            inp = torch.tensor(new_inp).to(inp)
        else:
            raise ValueError(f"Invalid fusion value. Expected one of ['max_pooling', 'average_pooling', 'sum'], got "
                             f"'{self.fusion}' instead.")

        return inp

    def _get_predictions(self, hidden_states, attention_mask):
        hidden_states = self.fusion_layer(hidden_states, attention_mask, 1) # (B, H)

        # Food
        food_score_preds = self.food_score(hidden_states).squeeze(-1)  # (B)
        food_existence_preds = self.food_existence(hidden_states).squeeze(-1)  # (B)
        food_score_preds = self.sigmoid(food_score_preds) * self.std + self.mean
        food_existence_preds = self.sigmoid(food_existence_preds)

        # Service
        service_score_preds = self.service_score(hidden_states).squeeze(-1)  # (B)
        service_existence_preds = self.service_existence(hidden_states).squeeze(-1)  # (B)
        service_score_preds = self.sigmoid(service_score_preds) * self.std + self.mean
        service_existence_preds = self.sigmoid(service_existence_preds)

        # Price
        price_score_preds = self.price_score(hidden_states).squeeze(-1)  # (B)
        price_existence_preds = self.price_existence(hidden_states).squeeze(-1)  # (B)
        price_score_preds = self.sigmoid(price_score_preds) * self.std + self.mean
        price_existence_preds = self.sigmoid(price_existence_preds)

        return food_score_preds, food_existence_preds, \
               service_score_preds, service_existence_preds, \
               price_score_preds, price_existence_preds

    def _compute_losses(
        self,
        # Predictions
        food_score_preds, food_existence_preds,
        service_score_preds, service_existence_preds,
        price_score_preds, price_existence_preds,
        # Groundtruths
        food_score_label, food_existence_label,
        service_score_label, service_existence_label,
        price_score_label, price_existence_label,
    ):
        """Compute losses (including total loss) given loss weights"""

        # food_score loss
        if food_existence_label.any():
            food_score_loss = F.mse_loss(food_score_preds[food_existence_label],
                                         food_score_label[food_existence_label].float())
        else:
            food_score_loss = 0.0

        # food existence loss
        food_existence_loss = F.binary_cross_entropy(food_existence_preds, food_existence_label.float())

        # service_score loss
        if service_existence_label.any():
            service_score_loss = F.mse_loss(service_score_preds[service_existence_label],
                                            service_score_label[service_existence_label].float())
        else:
            service_score_loss = 0.0

        # service existence loss
        service_existence_loss = F.binary_cross_entropy(service_existence_preds, service_existence_label.float())

        # price_score loss
        if price_existence_label.any():
            price_score_loss = F.mse_loss(price_score_preds[price_existence_label],
                                          price_score_label[price_existence_label].float())
        else:
            price_score_loss = 0.0

        # price existence loss
        price_existence_loss = F.binary_cross_entropy(price_existence_preds, price_existence_label.float())

        # Total loss
        total_loss = 0
        for weight, loss in \
                zip(self.lambdas, [food_score_loss, food_existence_loss,
                                   service_score_loss, service_existence_loss,
                                   price_score_loss, price_existence_loss]):
            total_loss += weight * loss

        return {"total_loss": total_loss,
                "food_score_loss": food_score_loss, "food_existence_loss": food_existence_loss,
                "service_score_loss": service_score_loss, "service_existence_loss": service_existence_loss,
                "price_score_loss": price_score_loss, "price_existence_loss": price_existence_loss
                }

    def forward(self, input_ids, attention_mask=None, food_score_label=None, service_score_label=None,
                price_score_label=None, is_training=True, **kwargs):
        """Forward logic.

        input_ids : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Indices of input sequence tokens in the vocabulary (i.e.,
            encoded).
        attention_mask : torch.Tensor
            Tensor of shape (batch_size, sequence_length). Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``.
        poi_start : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of start indices of POI (person of interest) in the input
            sequence. `-1` if not present.
        poi_end : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of end indices (**inclusive**) of POI (person of
            interest) in the input sequence. `-1` if not present.
        street_start : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of start indices of street in the input sequence. `-1` if
            not present.
        street_end : torch.Tensor
            Tensor of shape (batch_size,). Groundtruth labels of end indices (**inclusive**) of street in the input
            sequence. `-1` if not present.
        has_poi : torch.Tensor (optional)
            Tensor of shape (batch_size,). Mask to indicate whether there is POI info in a input sequence. This can
            also be obtained by masking `poi_start != -1`.
        has_street : torch.Tensor (optional)
            Tensor of shape (batch_size,). Mask to indicate whether there is street info in a input sequence. This can
            also be obtained by masking `street_start != -1`.
        """
        # Base forward (feature extractor)
        hidden_states = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # (B, L, H)
        attention_mask = attention_mask.bool()

        # Calculate existence
        food_existence_label = (food_score_label != 0).bool()
        service_existence_label = (service_score_label != 0).bool()
        price_existence_label = (price_score_label != 0).bool()

        food_score_preds, food_existence_preds, \
            service_score_preds, service_existence_preds, \
            price_score_preds, price_existence_preds = \
            self._get_predictions(hidden_states, attention_mask)

        outp = {"food_score_preds": food_score_preds,
                "food_existence_preds": food_existence_preds,
                "service_score_preds": service_score_preds,
                "service_existence_preds": service_existence_preds,
                "price_score_preds": price_score_preds,
                "price_existence_preds": price_existence_preds,
                "attention_mask": attention_mask,  # (B, L)
                }

        # Get loss if training (i.e., some tensors are not provided)
        if is_training:
            # Compute loss
            losses = self._compute_losses(
                # Predictions
                food_score_preds, food_existence_preds,
                service_score_preds, service_existence_preds,
                price_score_preds, price_existence_preds,
                # Groundtruths
                food_score_label, food_existence_label,
                service_score_label, service_existence_label,
                price_score_label, price_existence_label,
            )
            outp["losses"] = losses

        return outp
