"""
Data loader for BIOCLITE dataset
Dataset structure: 24 PD patients + 16 healthy controls
Each row = one session/day, columns 4-11 = exercises
Each exercise = table with IMU data (50 Hz)
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import os

class BIOCLITEDataset:
    """Load and process BIOCLITE smartwatch dataset"""
    
    def __init__(self, data_path='data/raw/BBDD_BIOCLITE_v0.mat'):
        self.data_path = data_path
        self.raw_data = None
        self.fs = 50  # Sampling frequency (Hz)
        self.participants = []
        self.all_sessions = []
        
        # Column indices (1-based to 0-based)
        self.COL_GROUP = 0      # 0=healthy, 1=PD
        self.COL_ID = 1         # Participant ID
        self.COL_DAY = 2        # Day number
        self.COL_EXERCISES = slice(3, 11)  # Columns 4-11 (exercises 1-8)
        self.COL_UPDRS_AVAIL = 11  # UPDRS available flag
        self.COL_CONTEXT = 12   # 0=unsupervised, 1=initial sup, 2=final sup
        
    def load_data(self):
        """Load the .mat file"""
        print(f"Loading data from {self.data_path}...")
        
        # Try different loading methods
        try:
            # Method 1: Standard loadmat
            mat_data = sio.loadmat(self.data_path, struct_as_record=False, squeeze_me=True)
            if 'BBDD_BIOCLITE' in mat_data:
                self.raw_data = mat_data['BBDD_BIOCLITE']
            elif 'None' in mat_data:
                self.raw_data = mat_data['None']
            else:
                self.raw_data = mat_data
        except:
            # Method 2: Try with different parameters
            mat_data = sio.loadmat(self.data_path, struct_as_record=True, squeeze_me=False)
            self.raw_data = mat_data
        
        print(f"✅ Data loaded successfully!")
        print(f"  - Shape: {self.raw_data.shape if hasattr(self.raw_data, 'shape') else 'N/A'}")
        
        return self.raw_data
    
    def parse_sessions(self):
        """Parse all sessions from the dataset"""
        sessions = []
        
        # The data is organized as a table where each row is a session
        for row_idx in range(self.raw_data.shape[0]):
            row = self.raw_data[row_idx]
            
            session_info = {
                'group': self._extract_value(row, self.COL_GROUP),
                'participant_id': self._extract_value(row, self.COL_ID),
                'day': self._extract_value(row, self.COL_DAY),
                'context': self._extract_value(row, self.COL_CONTEXT),
                'updrs_available': self._extract_value(row, self.COL_UPDRS_AVAIL),
                'exercises': []
            }
            
            # Extract exercises (columns 4-11)
            for ex_idx in range(3, 11):
                exercise_data = self._extract_exercise_data(row, ex_idx)
                if exercise_data is not None and len(exercise_data) > 0:
                    session_info['exercises'].append(exercise_data)
            
            if session_info['participant_id'] is not None:
                sessions.append(session_info)
        
        self.all_sessions = sessions
        print(f"✅ Parsed {len(sessions)} sessions")
        return sessions
    
    def _extract_value(self, row, col_idx):
        """Extract a single value from a row"""
        try:
            if hasattr(row, '__getitem__'):
                val = row[col_idx]
                if hasattr(val, '__float__'):
                    return float(val)
                elif isinstance(val, str):
                    return val.strip()
                return val
        except:
            pass
        return None
    
    def _extract_exercise_data(self, row, col_idx):
        """Extract IMU data for an exercise"""
        try:
            if hasattr(row, '__getitem__'):
                ex_data = row[col_idx]
                
                # If it's a structured array or table
                if hasattr(ex_data, 'shape') and len(ex_data.shape) == 2:
                    # Expected columns: date, time, ms_elapsed, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, group, id, ex_num, updrs
                    return ex_data
        except:
            pass
        return None
    
    def extract_all_windows(self, window_size=128, step_size=32):
        """Extract sliding windows from all exercises"""
        all_windows = []
        all_labels = []
        all_metadata = []
        
        for session in self.all_sessions:
            label = 1 if session['group'] == 1 else 0  # 1=PD, 0=Healthy
            
            for ex_idx, exercise in enumerate(session['exercises']):
                if exercise is not None and exercise.shape[0] >= window_size:
                    # Extract IMU columns (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
                    # Columns: 3,4,5 = acc, 6,7,8 = gyr (0-based after date/time/ms)
                    imu_cols = [3, 4, 5, 6, 7, 8]
                    imu_data = exercise[:, imu_cols]
                    
                    # Extract windows
                    for start in range(0, len(imu_data) - window_size, step_size):
                        window = imu_data[start:start + window_size]
                        all_windows.append(window)
                        all_labels.append(label)
                        all_metadata.append({
                            'participant': session['participant_id'],
                            'day': session['day'],
                            'exercise': ex_idx + 1,
                            'context': session['context']
                        })
        
        return np.array(all_windows), np.array(all_labels), all_metadata
    
    def get_participant_split(self, test_participants, window_size=128, step_size=32):
        """Split data by participants for LOSO validation"""
        all_windows, all_labels, all_metadata = self.extract_all_windows(window_size, step_size)
        
        # Create masks
        test_mask = np.array([m['participant'] in test_participants for m in all_metadata])
        train_mask = ~test_mask
        
        return (all_windows[train_mask], all_labels[train_mask], 
                all_windows[test_mask], all_labels[test_mask])
    
    def get_summary(self):
        """Get dataset summary"""
        if not self.all_sessions:
            self.parse_sessions()
        
        participants = set()
        pd_sessions = 0
        healthy_sessions = 0
        
        for session in self.all_sessions:
            participants.add(session['participant_id'])
            if session['group'] == 1:
                pd_sessions += 1
            else:
                healthy_sessions += 1
        
        return {
            'total_participants': len(participants),
            'total_sessions': len(self.all_sessions),
            'pd_sessions': pd_sessions,
            'healthy_sessions': healthy_sessions,
            'sampling_rate': self.fs
        }


class IMUDataset(Dataset):
    """PyTorch Dataset for IMU data"""
    
    def __init__(self, sequences, labels, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create train and test dataloaders"""
    train_dataset = IMUDataset(X_train, y_train)
    test_dataset = IMUDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Test function
def test_load():
    """Test loading the dataset"""
    dataset = BIOCLITEDataset()
    dataset.load_data()
    summary = dataset.get_summary()
    
    print("\n📊 Dataset Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    
    return dataset

if __name__ == "__main__":
    test_load()
