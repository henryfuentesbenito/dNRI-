import torch
from torch.utils.data import DataLoader

import relg

model = relg.dnri(
    input_dim=4,
    encoder_hidden_dim=96,
    rnn_encoder_dim=96,
    decoder_dim=96,
    num_objects=3,
    num_edge_types=2,
    prior=[0.9, 0.1],
)

dataset = relg.SynthDataLoader("datasets", "train")
generator = relg.LoaderWrapper(
    DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
    ),
    512,
)

trainer = relg.Trainer(model, generator, epochs=100)
trainer.train()

torch.save(model.encoder, "models/encoder_09_prior")
