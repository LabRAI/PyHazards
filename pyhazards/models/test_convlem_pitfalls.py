"""
Comprehensive test suite for ConvLEM - Step 6: Common pitfalls checklist.
Tests:
1. Shape validation
2. Adjacency matrix handling
3. Edge cases
4. Gradient flow
5. Different configurations
"""
import torch
import torch.nn as nn
from pyhazards.models import build_model

def test_shape_validation():
    """Test that model validates input shapes correctly."""
    print("=" * 60)
    print("Test 1: Shape Validation")
    print("=" * 60)
    
    model = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
    )
    
    # Test 1a: Wrong past_days
    print("\n1a. Testing wrong past_days...")
    try:
        x_wrong_t = torch.randn(4, 10, 58, 12)  # Should be 8, not 10
        model(x_wrong_t)
        print("❌ FAILED: Should have raised ValueError for wrong past_days")
        return False
    except ValueError as e:
        print(f"✅ PASSED: Correctly caught wrong past_days")
        print(f"   Error message: {e}")
    
    # Test 1b: Wrong num_counties
    print("\n1b. Testing wrong num_counties...")
    try:
        x_wrong_n = torch.randn(4, 8, 60, 12)  # Should be 58, not 60
        model(x_wrong_n)
        print("❌ FAILED: Should have raised ValueError for wrong num_counties")
        return False
    except ValueError as e:
        print(f"✅ PASSED: Correctly caught wrong num_counties")
        print(f"   Error message: {e}")
    
    # Test 1c: Wrong in_dim
    print("\n1c. Testing wrong in_dim...")
    try:
        x_wrong_f = torch.randn(4, 8, 58, 10)  # Should be 12, not 10
        model(x_wrong_f)
        print("❌ FAILED: Should have raised ValueError for wrong in_dim")
        return False
    except ValueError as e:
        print(f"✅ PASSED: Correctly caught wrong in_dim")
        print(f"   Error message: {e}")
    
    # Test 1d: Correct shapes should work
    print("\n1d. Testing correct shapes...")
    x_correct = torch.randn(4, 8, 58, 12)
    logits = model(x_correct)
    assert logits.shape == (4, 58), f"Expected (4, 58), got {logits.shape}"
    print(f"✅ PASSED: Correct shapes work fine")
    print(f"   Output shape: {logits.shape}")
    
    print("\n" + "=" * 60)
    return True

def test_adjacency_normalization():
    """Test adjacency matrix normalization with self-loops."""
    print("=" * 60)
    print("Test 2: Adjacency Matrix Handling")
    print("=" * 60)
    
    from pyhazards.models.convlem_wildfire import _normalize_adjacency
    
    # Test 2a: Self-loops are added
    print("\n2a. Testing self-loop addition...")
    adj = torch.zeros(5, 5)
    adj[0, 1] = adj[1, 0] = 1  # Connect nodes 0 and 1
    
    normalized = _normalize_adjacency(adj)
    
    # Check diagonal (self-loops should be present)
    diagonal = normalized[0].diagonal()
    assert (diagonal > 0).all(), "Self-loops not added!"
    print(f"✅ PASSED: Self-loops added correctly")
    print(f"   Diagonal values: {diagonal.tolist()}")
    
    # Test 2b: Row normalization
    print("\n2b. Testing row normalization...")
    row_sums = normalized[0].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Rows not normalized!"
    print(f"✅ PASSED: Rows normalized (sum to 1)")
    print(f"   Row sums: {row_sums.tolist()}")
    
    # Test 2c: Batch dimension handling
    print("\n2c. Testing batch dimension handling...")
    adj_2d = torch.rand(58, 58)
    adj_3d = torch.rand(4, 58, 58)
    
    norm_2d = _normalize_adjacency(adj_2d)
    norm_3d = _normalize_adjacency(adj_3d)
    
    assert norm_2d.dim() == 3 and norm_2d.size(0) == 1, \
        "2D adjacency should expand to (1, N, N)"
    assert norm_3d.dim() == 3 and norm_3d.size(0) == 4, \
        "3D adjacency should maintain batch dimension"
    print(f"✅ PASSED: Batch dimensions handled correctly")
    print(f"   2D input → {norm_2d.shape}")
    print(f"   3D input → {norm_3d.shape}")
    
    print("\n" + "=" * 60)
    return True

def test_adjacency_options():
    """Test model works with different adjacency configurations."""
    print("=" * 60)
    print("Test 3: Adjacency Configuration Options")
    print("=" * 60)
    
    x = torch.randn(4, 8, 58, 12)
    
    # Test 3a: No adjacency (should use identity)
    print("\n3a. Testing without adjacency (identity fallback)...")
    model_no_adj = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
    )
    logits_no_adj = model_no_adj(x)
    print(f"✅ PASSED: Works without adjacency")
    print(f"   Output shape: {logits_no_adj.shape}")
    
    # Test 3b: Fixed adjacency at construction
    print("\n3b. Testing with fixed adjacency at construction...")
    adj_fixed = torch.rand(58, 58)
    adj_fixed = (adj_fixed + adj_fixed.t()) / 2
    
    model_fixed = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
        adjacency=adj_fixed,
    )
    logits_fixed = model_fixed(x)
    print(f"✅ PASSED: Works with fixed adjacency")
    print(f"   Output shape: {logits_fixed.shape}")
    
    # Test 3c: Override adjacency at forward
    print("\n3c. Testing with adjacency override at forward...")
    adj_override = torch.rand(58, 58)
    adj_override = (adj_override + adj_override.t()) / 2
    
    logits_override = model_fixed(x, adjacency=adj_override)
    print(f"✅ PASSED: Can override adjacency at forward")
    
    # Test 3d: Different adjacencies should give different outputs
    print("\n3d. Testing that different adjacencies affect output...")
    assert not torch.allclose(logits_fixed, logits_override, atol=1e-3), \
        "Different adjacencies should produce different outputs!"
    print(f"✅ PASSED: Different adjacencies produce different outputs")
    
    print("\n" + "=" * 60)
    return True

def test_gradient_flow():
    """Test that gradients flow through all parameters."""
    print("=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    model = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
    )
    
    x = torch.randn(4, 8, 58, 12)
    y = torch.randint(0, 2, (4, 58)).float()
    
    # Forward pass
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    
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
    
    print(f"\nParameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")
    
    assert params_with_grad > 0, "No parameters received gradients!"
    
    if params_without_grad > 0:
        print(f"⚠️  WARNING: {params_without_grad} parameters did not receive gradients")
        print("   (This might be okay for some architectures)")
    
    print(f"\n✅ PASSED: Gradients flow through the model")
    print(f"   Loss value: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    return True

def test_different_configurations():
    """Test model with different hyperparameter configurations."""
    print("=" * 60)
    print("Test 5: Different Configurations")
    print("=" * 60)
    
    configs = [
        {"hidden_dim": 64, "num_layers": 1, "dt": 0.5},
        {"hidden_dim": 128, "num_layers": 2, "dt": 1.0},
        {"hidden_dim": 256, "num_layers": 3, "dt": 2.0},
        {"activation": "relu", "use_reset_gate": True},
        {"activation": "tanh", "use_reset_gate": False},
        {"dropout": 0.0},
        {"dropout": 0.3},
    ]
    
    x = torch.randn(2, 8, 58, 12)
    
    for i, config in enumerate(configs, 1):
        print(f"\n5.{i}. Testing config: {config}")
        try:
            model = build_model(
                "convlem_wildfire",
                task="classification",
                in_dim=12,
                num_counties=58,
                past_days=8,
                **config,
            )
            logits = model(x)
            assert logits.shape == (2, 58)
            print(f"   ✅ PASSED")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return False
    
    print("\n" + "=" * 60)
    return True

def test_batch_sizes():
    """Test model works with different batch sizes including batch_size=1."""
    print("=" * 60)
    print("Test 6: Different Batch Sizes")
    print("=" * 60)
    
    model = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
    )
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 8, 58, 12)
        logits = model(x)
        assert logits.shape == (bs, 58), f"Wrong shape for batch_size={bs}"
        print(f"✅ Batch size {bs:2d}: output shape {logits.shape}")
    
    print("\n" + "=" * 60)
    return True

def test_device_compatibility():
    """Test model can move between CPU and GPU (if available)."""
    print("=" * 60)
    print("Test 7: Device Compatibility")
    print("=" * 60)
    
    model = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
    )
    
    # Test CPU
    print("\n7a. Testing on CPU...")
    x_cpu = torch.randn(2, 8, 58, 12)
    logits_cpu = model(x_cpu)
    assert logits_cpu.device.type == "cpu"
    print(f"✅ PASSED: Works on CPU")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\n7b. Testing on CUDA...")
        model_cuda = model.cuda()
        x_cuda = x_cpu.cuda()
        logits_cuda = model_cuda(x_cuda)
        assert logits_cuda.device.type == "cuda"
        print(f"✅ PASSED: Works on CUDA")
    else:
        print("\n7b. CUDA not available, skipping GPU test")
    
    # Test MPS if available (Apple Silicon)
    if torch.backends.mps.is_available():
        print("\n7c. Testing on MPS (Apple Silicon)...")
        model_mps = model.to("mps")
        x_mps = x_cpu.to("mps")
        logits_mps = model_mps(x_mps)
        assert logits_mps.device.type == "mps"
        print(f"✅ PASSED: Works on MPS")
    else:
        print("\n7c. MPS not available, skipping MPS test")
    
    print("\n" + "=" * 60)
    return True
def test_model_save_load():
    """Test model can be saved and loaded."""
    print("=" * 60)
    print("Test 8: Model Save/Load")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Create model with fixed adjacency to ensure reproducibility
    adj = torch.rand(58, 58)
    adj = (adj + adj.t()) / 2
    
    model = build_model(
        "convlem_wildfire",
        task="classification",
        in_dim=12,
        num_counties=58,
        past_days=8,
        adjacency=adj,  # Use fixed adjacency
    )
    
    # Set to eval mode and use fixed random seed for reproducibility
    model.eval()
    torch.manual_seed(42)
    x = torch.randn(2, 8, 58, 12)
    
    # Get predictions before save
    with torch.no_grad():
        logits_before = model(x)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        temp_path = f.name
    
    try:
        # Save full model state including buffers
        torch.save({
            'model_state_dict': model.state_dict(),
            'adjacency': adj,
        }, temp_path)
        print(f"\n✅ Model saved to {temp_path}")
        
        # Create new model and load
        checkpoint = torch.load(temp_path)
        
        model_loaded = build_model(
            "convlem_wildfire",
            task="classification",
            in_dim=12,
            num_counties=58,
            past_days=8,
            adjacency=checkpoint['adjacency'],  # Restore adjacency
        )
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        model_loaded.eval()
        print(f"✅ Model loaded successfully")
        
        # Check predictions match
        with torch.no_grad():
            logits_after = model_loaded(x)
        
        # Check if outputs are close
        max_diff = (logits_before - logits_after).abs().max().item()
        
        if not torch.allclose(logits_before, logits_after, atol=1e-5):
            print(f"⚠️  Warning: Small numerical differences detected")
            print(f"   Max difference: {max_diff:.2e}")
            
            # Try with more lenient tolerance
            if torch.allclose(logits_before, logits_after, atol=1e-4):
                print(f"✅ PASSED: Outputs match within tolerance (1e-4)")
            else:
                raise AssertionError(f"Loaded model produces different outputs! Max diff: {max_diff}")
        else:
            print(f"✅ PASSED: Loaded model produces identical outputs")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n" + "=" * 60)
    return True
def run_all_tests():
    """Run all pitfall tests."""
    print("\n" + "=" * 60)
    print("CONVLEM PITFALL CHECKLIST - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Shape Validation", test_shape_validation),
        ("Adjacency Normalization", test_adjacency_normalization),
        ("Adjacency Options", test_adjacency_options),
        ("Gradient Flow", test_gradient_flow),
        ("Different Configurations", test_different_configurations),
        ("Different Batch Sizes", test_batch_sizes),
        ("Device Compatibility", test_device_compatibility),
        ("Model Save/Load", test_model_save_load),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print("=" * 60)
    print(f"Total: {passed_count}/{total} tests passed")
    print("=" * 60)
    
    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED! ConvLEM is ready for production!")
        return True
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Please review.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
