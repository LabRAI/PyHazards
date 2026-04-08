# test_dlwp_registration.py

from pyhazards.models import available_models, build_model
import torch

# Check registration
print("Available models:", available_models())
assert "dlwp" in available_models(), "DLWP not registered!"
print("✅ DLWP is registered")

# Test building
model = build_model(
    name="dlwp",
    task="regression",
    in_channels=7,
    num_faces=6,
    face_size=32,  # Small for testing
)
print(f"✅ Model built successfully")
print(f"   Model type: {type(model).__name__}")

# Test forward pass
x = torch.randn(2, 7, 6, 32, 32)
y = model(x)
print(f"✅ Forward pass successful")
print(f"   Input shape:  {tuple(x.shape)}")
print(f"   Output shape: {tuple(y.shape)}")
assert y.shape == (2, 7, 6, 32, 32), f"Wrong output shape: {y.shape}"

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Total parameters: {total_params:,}")

print("\n🎉 Step 4 (Registration) COMPLETE!")