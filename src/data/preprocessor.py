"""
Preprocessing module for BIOCLITE CSV data
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import RobustScaler, StandardScaler
import pywt

class IMUPreprocessor:
    """Preprocessing for IMU signals from CSV data"""
    
    def __init__(self, fs=50):
        self.fs = fs
        self.scaler = RobustScaler()
        
    def butter_bandpass(self, data, lowcut=0.5, highcut=20, order=4):
        """Apply bandpass filter"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)
    
    def remove_gravity(self, acc_data, cutoff=0.3):
        """Remove gravity component from accelerometer using high-pass filter"""
        nyquist = 0.5 * self.fs
        cutoff_norm = cutoff / nyquist
        b, a = signal.butter(4, cutoff_norm, btype='high')
        return signal.filtfilt(b, a, acc_data, axis=0)
    
    def normalize(self, data, fit=False):
        """Normalize data using RobustScaler"""
        original_shape = data.shape
        if len(original_shape) == 2:
            if fit:
                data = self.scaler.fit_transform(data)
            else:
                data = self.scaler.transform(data)
        else:
            data_flat = data.reshape(-1, data.shape[-1])
            if fit:
                data_flat = self.scaler.fit_transform(data_flat)
            else:
                data_flat = self.scaler.transform(data_flat)
            data = data_flat.reshape(original_shape)
        return data
    
    def extract_features(self, acc_window, gyro_window):
        """Extract comprehensive features from ACC and GYRO windows"""
        features = {}
        
        # Magnitude
        acc_mag = np.sqrt(np.sum(acc_window ** 2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro_window ** 2, axis=1))
        
        # Time-domain features
        for name, sig in [('acc', acc_mag), ('gyro', gyro_mag)]:
            features[f'{name}_mean'] = np.mean(sig)
            features[f'{name}_std'] = np.std(sig)
            features[f'{name}_rms'] = np.sqrt(np.mean(sig ** 2))
            features[f'{name}_max'] = np.max(sig)
            features[f'{name}_min'] = np.min(sig)
            features[f'{name}_range'] = np.max(sig) - np.min(sig)
            features[f'{name}_skew'] = skew(sig)
            features[f'{name}_kurtosis'] = kurtosis(sig)
        
        # Frequency-domain features
        for name, sig in [('acc', acc_mag), ('gyro', gyro_mag)]:
            freqs, psd = signal.welch(sig, fs=self.fs, nperseg=min(64, len(sig)))
            
            # Band powers
            bands = {'bradykinesia': (0.5, 3), 'tremor': (3, 8), 'high': (8, 12)}
            total_power = np.sum(psd) + 1e-9
            
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[mask])
                features[f'{name}_power_{band_name}'] = band_power
                features[f'{name}_rel_power_{band_name}'] = band_power / total_power
            
            # Dominant frequency
            features[f'{name}_dom_freq'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        
        # Jerk (derivative of acceleration)
        jerk = np.diff(acc_mag)
        if len(jerk) > 0:
            features['jerk_mean'] = np.mean(jerk)
            features['jerk_std'] = np.std(jerk)
            features['jerk_rms'] = np.sqrt(np.mean(jerk ** 2))
        
        # Zero-crossing rate
        features['zcr_acc'] = np.sum(acc_mag[:-1] * acc_mag[1:] < 0) / len(acc_mag)
        features['zcr_gyro'] = np.sum(gyro_mag[:-1] * gyro_mag[1:] < 0) / len(gyro_mag)
        
        return features
    
    def extract_features_batch(self, acc_windows, gyro_windows):
        """Extract features from batch of windows"""
        features_list = []
        for acc_win, gyro_win in zip(acc_windows, gyro_windows):
            feats = self.extract_features(acc_win, gyro_win)
            features_list.append(list(feats.values()))
        return np.array(features_list)