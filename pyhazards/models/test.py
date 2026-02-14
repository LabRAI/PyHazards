import torch
from pyhazards.datasets import GraphTemporalDataset, graph_collate, DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model

# Setup
past_days, counties, feats, samples = 8, 58, 12, 64

# Generate dummy data
x = torch.randn(samples, past_days, counties, feats)
y = torch.randint(0, 2, (samples, counties)).float()
adj = torch.rand(counties, counties)
adj = (adj + adj.t()) / 2  # Make symmetric

# Create datasets
train_ds = GraphTemporalDataset(x[:48], y[:48], adjacency=adj)
val_ds = GraphTemporalDataset(x[48:], y[48:], adjacency=adj)

# Create DataBundle
bundle = DataBundle(
    splits={"train": DataSplit(train_ds, None), "val": DataSplit(val_ds, None)},
    feature_spec=FeatureSpec(input_dim=feats, extra={"past_days": past_days, "counties": counties}),
    label_spec=LabelSpec(num_targets=counties, task_type="classification"),
)

# Build model
model = build_model(
    name="convlem_wildfire",
    task="classification",
    in_dim=feats,
    num_counties=counties,
    past_days=past_days,
    adjacency=adj,
)

# Setup trainer
trainer = Trainer(model=model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Train
print("Starting training...")
trainer.fit(
    bundle, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    max_epochs=2, 
    batch_size=8, 
    collate_fn=graph_collate
)

# Test inference
print("\nTesting inference...")
with torch.no_grad():
    test_x = x[:1]
    logits = model(test_x, adjacency=adj)
    probs = torch.sigmoid(logits)
    print(f"✓ Inference works! Probabilities shape: {probs.shape}")
    print(f"✓ Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

print("\n✅ ConvLEM training test PASSED!")