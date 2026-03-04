"""
Test data format compatibility with WaveCastNet.
"""
import torch
from torch.utils.data import TensorDataset
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model

def test_simple_tensor_format():
    """Test with simple tensor datasets (easiest approach)."""
    print("=" * 60)
    print("Test: Simple Tensor Format")
    print("=" * 60)
    
    # Generate synthetic wavefield data
    # In practice, load from .npy, .pt, or .h5 files
    num_samples = 32
    
    # Use small dimensions for testing
    x_train = torch.randn(num_samples, 3, 10, 64, 64)  # Past wavefields
    y_train = torch.randn(num_samples, 3, 10, 64, 64)  # Future wavefields
    
    x_val = torch.randn(8, 3, 10, 64, 64)
    y_val = torch.randn(8, 3, 10, 64, 64)
    
    print(f"✓ Data shapes:")
    print(f"  Train input:  {tuple(x_train.shape)}")
    print(f"  Train target: {tuple(y_train.shape)}")
    print(f"  Val input:    {tuple(x_val.shape)}")
    print(f"  Val target:   {tuple(y_val.shape)}")
    
    # Create simple tensor datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    # Wrap in PyHazards DataBundle
    # NOTE: DataSplit signature is DataSplit(dataset, transform)
    bundle = DataBundle(
        splits={
            "train": DataSplit(train_dataset, None),  # None = no transform
            "val": DataSplit(val_dataset, None),
        },
        feature_spec=FeatureSpec(
            input_dim=3,
            extra={"temporal_in": 10, "temporal_out": 10, "height": 64, "width": 64}
        ),
        label_spec=LabelSpec(
            num_targets=3,
            task_type="regression"
        ),
    )
    
    print(f"✓ DataBundle created")
    
    # Build model
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=64,
        width=64,
        temporal_in=10,
        temporal_out=10,
    )
    
    print(f"✓ Model built")
    
    # Setup training
    trainer = Trainer(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    
    print(f"✓ Trainer configured")
    print(f"\nStarting training...")
    
    # Train for 2 epochs
    # NOTE: collate_fn goes in trainer.fit(), not DataSplit
    trainer.fit(
        bundle,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=2,
        batch_size=4,
    )
    
    print(f"\n✓ Training completed successfully")
    
    # Test inference
    print(f"\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 10, 64, 64)
        test_output = model(test_input)
        assert test_output.shape == (1, 3, 10, 64, 64), \
            f"Wrong output shape: {test_output.shape}"
    
    print(f"✓ Inference works")
    print(f"  Input:  {tuple(test_input.shape)}")
    print(f"  Output: {tuple(test_output.shape)}")
    
    print("\n" + "=" * 60)
    print("✅ Step 5 (Data Format Matching) COMPLETE!")
    print("=" * 60)

def test_dataloader_compatibility():
    """Test that model works with standard PyTorch DataLoader."""
    print("\n" + "=" * 60)
    print("Test: Direct DataLoader Compatibility")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    # Create dataset
    x = torch.randn(16, 3, 10, 64, 64)
    y = torch.randn(16, 3, 10, 64, 64)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Build model
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=64,
        width=64,
        temporal_in=10,
        temporal_out=10,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    
    # Manual training loop
    model.train()
    for epoch in range(2):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print(f"\n✓ Direct DataLoader training works")
    print("=" * 60)

if __name__ == "__main__":
    # Test 1: PyHazards DataBundle format
    test_simple_tensor_format()
    
    # Test 2: Direct PyTorch DataLoader
    test_dataloader_compatibility()
    
    print("\n🎉 All data format tests passed!")
