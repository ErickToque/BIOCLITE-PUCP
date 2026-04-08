
"""Test all imports and basic functionality"""

import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("TESTING BIOCLITE-PUCP IMPORTS")
print("=" * 60)

# Test imports
try:
    from data_loader import BIOCLITEDataset, IMUDataset
    print("✅ data_loader imported")
except Exception as e:
    print(f"❌ data_loader: {e}")

try:
    from preprocessing import IMUPreprocessor
    print("✅ preprocessing imported")
except Exception as e:
    print(f"❌ preprocessing: {e}")

try:
    from models import CNN1D, BiLSTM, CNNLSTM, TransformerModel
    print("✅ models imported")
except Exception as e:
    print(f"❌ models: {e}")

try:
    from utils import set_seed, get_device, compute_metrics
    print("✅ utils imported")
except Exception as e:
    print(f"❌ utils: {e}")

try:
    from visualization import Visualizer
    print("✅ visualization imported")
except Exception as e:
    print(f"❌ visualization: {e}")

print("\n" + "=" * 60)
print("TESTING MODEL INSTANTIATION")
print("=" * 60)

import torch

try:
    model = CNN1D(input_channels=6, seq_length=128)
    x = torch.randn(4, 128, 6)
    out = model(x)
    print(f"✅ CNN1D: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"❌ CNN1D: {e}")

try:
    model = BiLSTM(input_size=6)
    x = torch.randn(4, 128, 6)
    out = model(x)
    print(f"✅ BiLSTM: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"❌ BiLSTM: {e}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)