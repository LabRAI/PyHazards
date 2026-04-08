"""
Training script for DLWP using PyHazards Trainer.

This demonstrates proper training logic porting:
- Model logic stays in nn.Module
- Use PyHazards Trainer for training loop
- Standard PyTorch losses (MSE)
- Custom weather metrics computed during evaluation
"""

import torch
from torch.utils.data import TensorDataset
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model
from pyhazards.models.dlwp import WeatherMetrics


def create_synthetic_data(num_samples=100, channels=7, faces=6, size=32):
    """Create synthetic weather data for demonstration."""
    # Current states
    x = torch.randn(num_samples, channels, faces, size, size)
    
    # Future states (with some correlation to current)
    # In reality, this would be actual ERA5 data
    y = x + 0.1 * torch.randn(num_samples, channels, faces, size, size)
    
    return x, y


def main():
    print("=" * 70)
    print("DLWP Training with PyHazards")
    print("=" * 70)
    
    # ===== 1. PREPARE DATA =====
    print("\n1. Preparing data...")
    
    x_train, y_train = create_synthetic_data(num_samples=80)
    x_val, y_val = create_synthetic_data(num_samples=20)
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    bundle = DataBundle(
        splits={
            "train": DataSplit(train_dataset, None),
            "val": DataSplit(val_dataset, None),
        },
        feature_spec=FeatureSpec(
            input_dim=7,
            extra={"num_faces": 6, "face_size": 32}
        ),
        label_spec=LabelSpec(num_targets=7, task_type="regression"),
    )
    
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Val samples:   {len(val_dataset)}")
    print(f"   ✓ Input shape:   {tuple(x_train[0].shape)}")
    
    # ===== 2. BUILD MODEL =====
    print("\n2. Building model...")
    
    model = build_model(
        name="dlwp",
        task="regression",
        in_channels=7,
        num_faces=6,
        face_size=32,
        num_levels=3,      # Smaller for faster training
        base_channels=32,  # Smaller for faster training
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model: DLWP")
    print(f"   ✓ Parameters: {total_params:,}")
    
    # ===== 3. SETUP TRAINING =====
    print("\n3. Setting up training...")
    
    # Standard MSE loss (common for weather prediction)
    loss_fn = torch.nn.MSELoss()
    
    # Optimizer (Adam is standard)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(model=model)
    
    print(f"   ✓ Loss: MSELoss")
    print(f"   ✓ Optimizer: Adam (lr=1e-3)")
    
    # ===== 4. TRAIN =====
    print("\n4. Training...")
    print("-" * 70)
    
    trainer.fit(
        bundle,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=5,
        batch_size=8,
    )
    
    print("-" * 70)
    print("   ✓ Training complete")
    
    # ===== 5. EVALUATE =====
    print("\n5. Evaluating...")
    
    model.eval()
    with torch.no_grad():
        # Get validation predictions
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False
        )
        
        all_preds = []
        all_targets = []
        
        for inputs, targets in val_loader:
            preds = model(inputs)
            all_preds.append(preds)
            all_targets.append(targets)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute weather metrics
        metrics = WeatherMetrics.compute_all(all_preds, all_targets)
        
        print(f"\n   Validation Metrics:")
        print(f"   ✓ RMSE:  {metrics['RMSE']:.6f}")
        print(f"   ✓ Bias:  {metrics['Bias']:.6f}")
        print(f"   ✓ ACC:   {metrics['ACC']:.6f}")
    
    # ===== 6. AUTOREGRESSIVE FORECAST =====
    print("\n6. Testing autoregressive forecasting...")
    
    # Take one sample
    initial_state = x_val[0:1]  # (1, 7, 6, 32, 32)
    
    # Forecast 24 hours (4 steps × 6 hours)
    states = [initial_state]
    with torch.no_grad():
        for step in range(4):
            next_state = model(states[-1])
            states.append(next_state)
    
    print(f"   ✓ Generated 24-hour forecast:")
    print(f"     t+0hr  (initial)")
    print(f"     t+6hr  (step 1)")
    print(f"     t+12hr (step 2)")
    print(f"     t+18hr (step 3)")
    print(f"     t+24hr (step 4)")
    
    # ===== 7. SAVE MODEL =====
    print("\n7. Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }, 'dlwp_checkpoint.pt')
    
    print(f"   ✓ Saved to: dlwp_checkpoint.pt")
    
    print("\n" + "=" * 70)
    print("✅ Step 6 (Training Logic) COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()