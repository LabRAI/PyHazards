"""
Test data format compatibility with DLWP.
"""

import torch
from torch.utils.data import TensorDataset
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model


def test_simple_tensor_format():
    """Test with simple tensor datasets (easiest approach)."""
    print("=" * 70)
    print("Test: Simple Tensor Format for DLWP")
    print("=" * 70)
    
    # Generate synthetic weather data
    # In practice, load from HDF5 files
    num_samples = 32
    
    # Use small dimensions for testing
    # Shape: (samples, channels, faces, height, width)
    x_train = torch.randn(num_samples, 7, 6, 32, 32)  # Current states
    y_train = torch.randn(num_samples, 7, 6, 32, 32)  # Future states (+6hr)
    
    x_val = torch.randn(8, 7, 6, 32, 32)
    y_val = torch.randn(8, 7, 6, 32, 32)
    
    print(f"✓ Data shapes:")
    print(f"  Train input:  {tuple(x_train.shape)}")
    print(f"  Train target: {tuple(y_train.shape)}")
    print(f"  Val input:    {tuple(x_val.shape)}")
    print(f"  Val target:   {tuple(y_val.shape)}")
    
    # Create simple tensor datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    # Wrap in PyHazards DataBundle
    bundle = DataBundle(
        splits={
            "train": DataSplit(train_dataset, None),
            "val": DataSplit(val_dataset, None),
        },
        feature_spec=FeatureSpec(
            input_dim=7,
            extra={"num_faces": 6, "face_size": 32}
        ),
        label_spec=LabelSpec(
            num_targets=7,
            task_type="regression"
        ),
    )
    
    print(f"✓ DataBundle created")
    
    # Build model
    model = build_model(
        name="dlwp",
        task="regression",
        in_channels=7,
        num_faces=6,
        face_size=32,
    )
    
    print(f"✓ Model built")
    
    # Setup training
    trainer = Trainer(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    print(f"✓ Trainer configured")
    print(f"\nStarting training...")
    
    # Train for 2 epochs
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
        test_input = torch.randn(1, 7, 6, 32, 32)
        test_output = model(test_input)
        assert test_output.shape == (1, 7, 6, 32, 32), \
            f"Wrong output shape: {test_output.shape}"
    
    print(f"✓ Inference works")
    print(f"  Input:  {tuple(test_input.shape)}")
    print(f"  Output: {tuple(test_output.shape)}")
    
    print("\n" + "=" * 70)
    print("✅ Step 5 (Data Format Matching) COMPLETE!")
    print("=" * 70)


def test_autoregressive_forecasting():
    """Test multi-step autoregressive forecasting (24hr = 4 × 6hr)."""
    print("\n" + "=" * 70)
    print("Test: Autoregressive Multi-Step Forecasting")
    print("=" * 70)
    
    # Build model
    model = build_model(
        name="dlwp",
        task="regression",
        in_channels=7,
        num_faces=6,
        face_size=32,
    )
    
    model.eval()
    
    # Initial state
    current_state = torch.randn(1, 7, 6, 32, 32)
    print(f"\nInitial state shape: {tuple(current_state.shape)}")
    
    # Forecast 24 hours ahead (4 steps of 6hr each)
    states = [current_state]
    
    with torch.no_grad():
        for step in range(4):
            next_state = model(states[-1])
            states.append(next_state)
            print(f"  Step {step+1} (+{(step+1)*6}hr): {tuple(next_state.shape)}")
    
    print(f"\n✓ Generated {len(states)} states:")
    print(f"  states[0] = t+0hr  (initial)")
    print(f"  states[1] = t+6hr")
    print(f"  states[2] = t+12hr")
    print(f"  states[3] = t+18hr")
    print(f"  states[4] = t+24hr")
    
    # Stack into sequence
    forecast_sequence = torch.cat(states, dim=0)
    print(f"\n✓ Full forecast sequence: {tuple(forecast_sequence.shape)}")
    
    print("=" * 70)


def test_dataloader_compatibility():
    """Test that model works with standard PyTorch DataLoader."""
    print("\n" + "=" * 70)
    print("Test: Direct DataLoader Compatibility")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    
    # Create dataset
    x = torch.randn(16, 7, 6, 32, 32)
    y = torch.randn(16, 7, 6, 32, 32)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Build model
    model = build_model(
        name="dlwp",
        task="regression",
        in_channels=7,
        num_faces=6,
        face_size=32,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
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
    print("=" * 70)


if __name__ == "__main__":
    # Test 1: PyHazards DataBundle format
    test_simple_tensor_format()
    
    # Test 2: Autoregressive forecasting
    test_autoregressive_forecasting()
    
    # Test 3: Direct PyTorch DataLoader
    test_dataloader_compatibility()
    
    print("\n🎉 All data format tests passed!")