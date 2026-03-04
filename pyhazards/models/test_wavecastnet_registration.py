from pyhazards.models import available_models, build_model
import torch

# Check registration
print("Available models:", available_models())
assert "wavecastnet" in available_models(), "WaveCastNet not registered!"
print("✅ WaveCastNet is registered")

# Test building with SMALLER dimensions for testing
print("\nBuilding model with test dimensions...")
model = build_model(
    name="wavecastnet",
    task="regression",
    in_channels=3,
    height=64,        # Smaller: was 344
    width=64,         # Smaller: was 224
    temporal_in=10,   # Shorter: was 60
    temporal_out=10,  # Shorter: was 60
)
print(f"✅ Model built successfully")
print(f"   Model type: {type(model).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Total parameters: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Test forward pass with small batch
print("\nTesting forward pass...")
x = torch.randn(1, 3, 10, 64, 64)  # Small test batch
print(f"   Input shape:  {tuple(x.shape)}")

y = model(x)
print(f"✅ Forward pass successful")
print(f"   Output shape: {tuple(y.shape)}")
assert y.shape == (1, 3, 10, 64, 64), f"Wrong output shape: {y.shape}"

# Test with batch size 2
print("\nTesting with batch_size=2...")
x2 = torch.randn(2, 3, 10, 64, 64)
y2 = model(x2)
print(f"✅ Batch processing works")
print(f"   Output shape: {tuple(y2.shape)}")

# Test shape validation
print("\nTesting shape validation...")
try:
    x_wrong = torch.randn(2, 3, 15, 64, 64)  # Wrong temporal_in
    model(x_wrong)
    print("❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Shape validation works: {e}")

print("\n" + "="*60)
print("🎉 Step 4 (Registration) COMPLETE!")
print("="*60)
print("\nNote: For production use with full spatial resolution (344×224)")
print("and full temporal sequences (60 timesteps), you'll need:")
print("  - GPU with sufficient VRAM (8GB+)")
print("  - Or smaller batch sizes")
print("  - Or gradient checkpointing for memory efficiency")
