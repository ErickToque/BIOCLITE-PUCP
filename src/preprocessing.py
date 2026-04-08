"""
Preprocessing module for BIOCLITE smartwatch data
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
import pywt

class IMUPreprocessor:
    """Preprocessing for IMU signals (accelerometer + gyroscope)"""
    
    def __init__(self, fs=50):
        self.fs = fs
        self.scaler = StandardScaler()
        
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
    
    def normalize(self, data):
        """Normalize data using StandardScaler"""
        original_shape = data.shape
        if len(original_shape) == 2:
            data = self.scaler.fit_transform(data)
        else:
            # For 3D data (windows)
            data_flat = data.reshape(-1, data.shape[-1])
            data_flat = self.scaler.fit_transform(data_flat)
            data = data_flat.reshape(original_shape)
        return data
    
    def extract_features(self, window):
        """Extract statistical and frequency features from window"""
        features = []
        
        for channel in range(window.shape[1]):
            data = window[:, channel]
            
            # Time domain features
            features.extend([
                np.mean(data),
                np.std(data),
                np.max(data),
                np.min(data),
                np.median(data),
                np.percentile(data, 25),
                np.percentile(data, 75),
                np.sum(data**2) / len(data)  # Energy
            ])
            
            # Frequency domain features
            fft_vals = np.abs(fft(data))[:len(data)//2]
            freqs = fftfreq(len(data), 1/self.fs)[:len(data)//2]
            
            if len(fft_vals) > 1:
                dominant_idx = np.argmax(fft_vals[1:]) + 1
                features.append(freqs[dominant_idx])
                features.append(fft_vals[dominant_idx])
                
                # Power in frequency bands
                mask_low = (freqs >= 0) & (freqs < 0.5)
                mask_walk = (freqs >= 0.5) & (freqs <= 3)
                features.append(np.sum(fft_vals[mask_low]))
                features.append(np.sum(fft_vals[mask_walk]))
            else:
                features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def extract_wavelet_features(self, window, wavelet='db4', level=3):
        """Extract wavelet features"""
        features = []
        
        for channel in range(window.shape[1]):
            data = window[:, channel]
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            for coeff in coeffs:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.sum(coeff**2)  # Energy
                ])
        
        return np.array(features)
