# transformer_model.py

import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, src_input_dim, tgt_input_dim, d_model, nhead, num_layers, dropout
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.src_input_linear = nn.Linear(src_input_dim, d_model)
        self.tgt_input_linear = nn.Linear(tgt_input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,  # Set batch_first to True
        )
        self.output_linear = nn.Linear(
            d_model, 1
        )  # Assuming we're predicting one output feature

    def forward(self, src, tgt):
        src = self.src_input_linear(src)
        tgt = self.tgt_input_linear(tgt)

        # No need to permute dimensions since batch_first=True
        output = self.transformer(src, tgt)

        output = self.output_linear(output)
        return output
