"""
Training script for WaveCastNet using PyHazards Trainer.
This demonstrates proper training logic porting:
- Model logic stays in nn.Module
- Use PyHazards Trainer for training loop
- Custom loss defined separately
- Metrics computed during evaluation
"""
import torch
from torch.utils.data import TensorDataset
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model
from pyhazards.models.wavecastnet import WaveCastNetLoss, WavefieldMetrics

def create_synthetic_data(num_samples=100, temporal=10, height=64, width=64):
    """Create synthetic wavefield data for demonstration."""
    x = torch.randn(num_samples, 3, temporal, height, width)
    y = torch.randn(num_samples, 3, temporal, height, width)
    return x, y

def main():
    print("=" * 70)
    print("WaveCastNet Training with PyHazards")
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
            input_dim=3,
            extra={"temporal_in": 10, "temporal_out": 10, "height": 64, "width": 64}
        ),
        label_spec=LabelSpec(num_targets=3, task_type="regression"),
    )
    
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Val samples:   {len(val_dataset)}")
    print(f"   ✓ Input shape:   {tuple(x_train[0].shape)}")
    
    # ===== 2. BUILD MODEL =====
    print("\n2. Building model...")
    
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=64,
        width=64,
        temporal_in=10,
        temporal_out=10,
        hidden_dim=64,    # Smaller for faster training
        num_layers=2,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model: WaveCastNet")
    print(f"   ✓ Parameters: {total_params:,}")
    
    # ===== 3. SETUP TRAINING =====
    print("\n3. Setting up training...")
    
    # Custom loss from paper
    loss_fn = WaveCastNetLoss(delta=1.0)
    
    # Optimizer (paper uses Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(model=model)
    
    print(f"   ✓ Loss: Huber Loss (δ=1.0)")
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
        
        # Compute paper metrics
        metrics = WavefieldMetrics.compute_all(all_preds, all_targets)
        
        print(f"\n   Validation Metrics:")
        print(f"   ✓ ACC (Accuracy):  {metrics['ACC']:.4f}")
        print(f"   ✓ RFNE (Rel Error): {metrics['RFNE']:.4f}")
    
    # ===== 6. SAVE MODEL =====
    print("\n6. Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }, 'wavecastnet_checkpoint.pt')
    
    print(f"   ✓ Saved to: wavecastnet_checkpoint.pt")
    
    print("\n" + "=" * 70)
    print("✅ Step 6 (Training Logic) COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
