"""
Smoke test for WaveCastNet - verifies basic functionality works.

This test should:
1. Build model from registry
2. Execute forward pass with correct shapes
3. Run short training loop
4. Verify gradient flow
5. Test model save/load
6. Complete in < 1 minute

Run with: python3 test_wavecastnet_smoke.py
"""

import torch
from torch.utils.data import TensorDataset
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer
from pyhazards.models import build_model, available_models
from pyhazards.models.wavecastnet import WaveCastNetLoss, WavefieldMetrics
import tempfile
import os


def test_1_registry():
    """Test 1: Model is registered and can be built."""
    print("=" * 70)
    print("Test 1: Model Registration")
    print("=" * 70)
    
    # Check model is registered
    models = available_models()
    assert "wavecastnet" in models, "WaveCastNet not in registry!"
    print(f"✓ WaveCastNet found in registry")
    print(f"  Available models: {models}")
    
    # Build from registry
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
    )
    
    assert model is not None, "Failed to build model"
    assert hasattr(model, 'forward'), "Model missing forward method"
    print(f"✓ Model built successfully")
    print(f"  Model type: {type(model).__name__}")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    print()
    return True


def test_2_forward_pass():
    """Test 2: Forward pass with various input shapes."""
    print("=" * 70)
    print("Test 2: Forward Pass")
    print("=" * 70)
    
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
    )
    
    # Test single sample
    print("Testing batch_size=1...")
    x1 = torch.randn(1, 3, 5, 32, 32)
    y1 = model(x1)
    assert y1.shape == (1, 3, 5, 32, 32), f"Wrong output shape: {y1.shape}"
    print(f"✓ Single sample: {tuple(x1.shape)} → {tuple(y1.shape)}")
    
    # Test batch
    print("Testing batch_size=4...")
    x4 = torch.randn(4, 3, 5, 32, 32)
    y4 = model(x4)
    assert y4.shape == (4, 3, 5, 32, 32), f"Wrong output shape: {y4.shape}"
    print(f"✓ Batch:         {tuple(x4.shape)} → {tuple(y4.shape)}")
    
    # Test different temporal lengths (if model supports it)
    print("Testing different configurations...")
    model2 = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=64,
        width=64,
        temporal_in=10,
        temporal_out=10,
    )
    x_large = torch.randn(2, 3, 10, 64, 64)
    y_large = model2(x_large)
    assert y_large.shape == (2, 3, 10, 64, 64)
    print(f"✓ Larger config: {tuple(x_large.shape)} → {tuple(y_large.shape)}")
    
    print()
    return True


def test_3_shape_validation():
    """Test 3: Input shape validation."""
    print("=" * 70)
    print("Test 3: Shape Validation")
    print("=" * 70)
    
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
    )
    
    # Wrong number of channels
    print("Testing wrong in_channels...")
    try:
        x_wrong = torch.randn(2, 5, 5, 32, 32)  # 5 channels instead of 3
        model(x_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    # Wrong temporal length
    print("Testing wrong temporal_in...")
    try:
        x_wrong = torch.randn(2, 3, 10, 32, 32)  # 10 timesteps instead of 5
        model(x_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    # Wrong spatial dimensions
    print("Testing wrong spatial dimensions...")
    try:
        x_wrong = torch.randn(2, 3, 5, 64, 64)  # 64x64 instead of 32x32
        model(x_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    print()
    return True


def test_4_training_loop():
    """Test 4: Short training loop with Trainer."""
    print("=" * 70)
    print("Test 4: Training Loop")
    print("=" * 70)
    
    # Create small synthetic dataset
    x_train = torch.randn(16, 3, 5, 32, 32)
    y_train = torch.randn(16, 3, 5, 32, 32)
    x_val = torch.randn(4, 3, 5, 32, 32)
    y_val = torch.randn(4, 3, 5, 32, 32)
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    bundle = DataBundle(
        splits={
            "train": DataSplit(train_dataset, None),
            "val": DataSplit(val_dataset, None),
        },
        feature_spec=FeatureSpec(input_dim=3, extra={"temporal_in": 5}),
        label_spec=LabelSpec(num_targets=3, task_type="regression"),
    )
    
    # Build model
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
        hidden_dim=32,  # Small for fast testing
    )
    
    # Setup training
    trainer = Trainer(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = WaveCastNetLoss(delta=1.0)
    
    print("Running 2 epochs...")
    
    # Train
    trainer.fit(
        bundle,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=2,
        batch_size=4,
    )
    
    print(f"✓ Training completed without errors")
    print()
    return True


def test_5_gradient_flow():
    """Test 5: Gradient flow through model."""
    print("=" * 70)
    print("Test 5: Gradient Flow")
    print("=" * 70)
    
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
    )
    
    # Forward pass
    x = torch.randn(2, 3, 5, 32, 32)
    y = torch.randn(2, 3, 5, 32, 32)
    
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad += 1
    
    print(f"Parameters with gradients:    {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")
    
    assert params_with_grad > 0, "No parameters received gradients!"
    print(f"✓ Gradients flow correctly")
    print(f"  Loss value: {loss.item():.4f}")
    
    print()
    return True


def test_6_loss_and_metrics():
    """Test 6: Custom loss and metrics."""
    print("=" * 70)
    print("Test 6: Loss and Metrics")
    print("=" * 70)
    
    # Test loss
    loss_fn = WaveCastNetLoss(delta=1.0)
    
    pred = torch.randn(2, 3, 5, 32, 32)
    target = torch.randn(2, 3, 5, 32, 32)
    
    loss = loss_fn(pred, target)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ WaveCastNetLoss works")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test metrics
    metrics = WavefieldMetrics.compute_all(pred, target)
    
    assert "ACC" in metrics, "Missing ACC metric"
    assert "RFNE" in metrics, "Missing RFNE metric"
    assert -1 <= metrics["ACC"] <= 1, "ACC out of range"
    assert metrics["RFNE"] >= 0, "RFNE should be non-negative"
    
    print(f"✓ WavefieldMetrics work")
    print(f"  ACC:  {metrics['ACC']:.4f}")
    print(f"  RFNE: {metrics['RFNE']:.4f}")
    
    print()
    return True


def test_7_save_load():
    """Test 7: Model save and load."""
    print("=" * 70)
    print("Test 7: Save and Load")
    print("=" * 70)
    
    # Build model
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,
        height=32,
        width=32,
        temporal_in=5,
        temporal_out=5,
    )
    
    model.eval()
    
    # Get predictions before save
    x = torch.randn(1, 3, 5, 32, 32)
    with torch.no_grad():
        pred_before = model(x)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        torch.save(model.state_dict(), temp_path)
        print(f"✓ Model saved to {temp_path}")
        
        # Load model
        model_loaded = build_model(
            name="wavecastnet",
            task="regression",
            in_channels=3,
            height=32,
            width=32,
            temporal_in=5,
            temporal_out=5,
        )
        model_loaded.load_state_dict(torch.load(temp_path))
        model_loaded.eval()
        print(f"✓ Model loaded successfully")
        
        # Check predictions match
        with torch.no_grad():
            pred_after = model_loaded(x)
        
        assert torch.allclose(pred_before, pred_after, atol=1e-5), \
            "Loaded model produces different outputs!"
        print(f"✓ Loaded model produces identical outputs")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print()
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("WAVECASTNET SMOKE TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        ("Model Registration", test_1_registry),
        ("Forward Pass", test_2_forward_pass),
        ("Shape Validation", test_3_shape_validation),
        ("Training Loop", test_4_training_loop),
        ("Gradient Flow", test_5_gradient_flow),
        ("Loss and Metrics", test_6_loss_and_metrics),
        ("Save and Load", test_7_save_load),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print("=" * 70)
    print(f"Total: {passed_count}/{total} tests passed")
    print("=" * 70)
    
    if passed_count == total:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("WaveCastNet is ready for Step 8 (Documentation)")
        return True
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
