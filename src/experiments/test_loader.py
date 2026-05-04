"""Test the updated data loader"""

import sys
sys.path.insert(0, 'src')

from data_loader import BIOCLITEDataset, create_dataloaders
from preprocessing import IMUPreprocessor
from utils import print_dataset_info, set_seed, get_device

def main():
    set_seed(42)
    print(f"Device: {get_device()}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING BIOCLITE DATA")
    print("="*60)
    
    loader = BIOCLITEDataset(data_path='data/raw/BIOCLITE_data_v2.csv')
    df = loader.load_data()
    
    # Print info
    print_dataset_info(df)
    
    # Test exercise 6 (bradykinesia)
    print("\n" + "="*60)
    print("EXERCISE 6 - BRADYKINESIA (Foot Tapping)")
    print("="*60)
    
    df_ej6 = loader.get_exercise_data(exercise_num=6)
    print(f"Exercise 6 samples: {len(df_ej6):,}")
    
    # Extract windows
    X, y, groups = loader.extract_windows(df_ej6, window_size=100, step_size=50)
    print(f"\nWindows extracted: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Unique subjects: {len(np.unique(groups))}")
    
    # Create dataloaders
    train_loader, test_loader, scaler = create_dataloaders(
        X, y, groups, batch_size=32, train_ratio=0.7
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test preprocessor
    preprocessor = IMUPreprocessor(fs=50)
    
    # Extract features from first window
    acc_cols = ['Acc_X', 'Acc_Y', 'Acc_Z']
    gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    # Get a sample window
    sample_idx = 0
    sample_window = X[sample_idx]
    n_samples = sample_window.shape[0]
    
    acc_sample = sample_window[:, :3]
    gyro_sample = sample_window[:, 3:6]
    
    features = preprocessor.extract_features(acc_sample, gyro_sample)
    print(f"\nExtracted {len(features)} features from sample window:")
    for i, (k, v) in enumerate(features.items()):
        print(f"  {k}: {v:.4f}")
        if i > 10:
            print(f"  ... and {len(features)-11} more")
            break
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    main()