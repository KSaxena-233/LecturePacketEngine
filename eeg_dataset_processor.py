import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path
import mne
from mne.io import read_raw_edf, read_raw_bdf
from mne.preprocessing import ICA
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDatasetProcessor:
    def __init__(self, data_dir: str = 'data/eeg_datasets'):
        """Initialize EEG dataset processor.
        
        Args:
            data_dir: Directory containing EEG datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
    def download_dataset(self, dataset_name: str, url: str):
        """Download an EEG dataset.
        
        Args:
            dataset_name: Name of the dataset
            url: URL to download the dataset
        """
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Download and extract dataset
        import requests
        import zipfile
        from tqdm import tqdm
        
        logger.info(f"Downloading {dataset_name} from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_path = dataset_dir / f"{dataset_name}.zip"
        with open(zip_path, 'wb') as f, tqdm(
            desc=dataset_name,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        # Extract dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        # Clean up zip file
        zip_path.unlink()
        
    def preprocess_raw_data(self, raw_data: mne.io.Raw) -> mne.io.Raw:
        """Preprocess raw EEG data.
        
        Args:
            raw_data: Raw EEG data from MNE
            
        Returns:
            Preprocessed EEG data
        """
        # Filter data
        raw_data.filter(l_freq=0.5, h_freq=50)
        
        # Remove power line noise
        raw_data.notch_filter(freqs=50)
        
        # Apply ICA for artifact removal
        ica = ICA(n_components=0.99, random_state=42)
        ica.fit(raw_data)
        
        # Remove eye movement artifacts
        ica.exclude = [0, 1]  # Adjust based on ICA components
        raw_data = ica.apply(raw_data)
        
        return raw_data
        
    def extract_features(self, raw_data: mne.io.Raw, 
                        window_size: float = 2.0) -> np.ndarray:
        """Extract features from preprocessed EEG data.
        
        Args:
            raw_data: Preprocessed EEG data
            window_size: Size of analysis window in seconds
            
        Returns:
            Feature matrix
        """
        # Get data and sampling frequency
        data = raw_data.get_data()
        sfreq = raw_data.info['sfreq']
        
        # Calculate window size in samples
        window_samples = int(window_size * sfreq)
        
        # Initialize feature list
        features = []
        
        # Process each window
        for i in range(0, data.shape[1] - window_samples, window_samples):
            window = data[:, i:i + window_samples]
            
            # Calculate power in each frequency band
            window_features = []
            for band_name, (low, high) in self.freq_bands.items():
                # Bandpass filter
                b, a = signal.butter(4, [low/(sfreq/2), high/(sfreq/2)], btype='band')
                filtered = signal.filtfilt(b, a, window)
                
                # Calculate power
                power = np.mean(filtered ** 2, axis=1)
                window_features.extend(power)
                
            # Add statistical features
            window_features.extend([
                np.mean(window, axis=1),  # Mean
                np.std(window, axis=1),   # Standard deviation
                signal.entropy(window, axis=1)  # Spectral entropy
            ])
            
            features.append(window_features)
            
        return np.array(features)
        
    def process_dataset(self, dataset_path: str, 
                       label_mapping: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Process an entire EEG dataset.
        
        Args:
            dataset_path: Path to the dataset
            label_mapping: Optional mapping of dataset labels to our states
            
        Returns:
            Tuple of (features, labels)
        """
        # Load dataset
        if dataset_path.endswith('.edf'):
            raw = read_raw_edf(dataset_path, preload=True)
        elif dataset_path.endswith('.bdf'):
            raw = read_raw_bdf(dataset_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
            
        # Preprocess data
        raw = self.preprocess_raw_data(raw)
        
        # Extract features
        features = self.extract_features(raw)
        
        # Get labels (implementation depends on dataset format)
        labels = self._extract_labels(raw, label_mapping)
        
        return features, labels
        
    def _extract_labels(self, raw: mne.io.Raw, 
                       label_mapping: Optional[Dict[str, str]]) -> np.ndarray:
        """Extract labels from raw data.
        
        Args:
            raw: Raw EEG data
            label_mapping: Optional mapping of dataset labels to our states
            
        Returns:
            Array of labels
        """
        # Implementation depends on dataset format
        # This is a placeholder that should be customized per dataset
        if label_mapping:
            # Map dataset labels to our states
            return np.array([label_mapping.get(l, 'unknown') for l in raw.annotations.description])
        else:
            # Use dataset labels directly
            return np.array(raw.annotations.description)
            
    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, 
                          output_path: str):
        """Save processed data for later use.
        
        Args:
            features: Feature matrix
            labels: Label array
            output_path: Path to save processed data
        """
        data = {
            'features': features,
            'labels': labels
        }
        joblib.dump(data, output_path)
        logger.info(f"Saved processed data to {output_path}")
        
    def load_processed_data(self, input_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed data.
        
        Args:
            input_path: Path to processed data
            
        Returns:
            Tuple of (features, labels)
        """
        data = joblib.load(input_path)
        return data['features'], data['labels']

# Example usage for DEAP dataset
def process_deap_dataset():
    processor = EEGDatasetProcessor()
    
    # Download DEAP dataset
    processor.download_dataset(
        'DEAP',
        'http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html'
    )
    
    # Process each subject's data
    for subject in range(1, 33):
        data_path = processor.data_dir / 'DEAP' / f's{subject:02d}.bdf'
        
        # Define label mapping for DEAP
        label_mapping = {
            'high_arousal_high_valence': 'engaged',
            'high_arousal_low_valence': 'confused',
            'low_arousal_high_valence': 'neutral',
            'low_arousal_low_valence': 'fatigued'
        }
        
        # Process data
        features, labels = processor.process_dataset(data_path, label_mapping)
        
        # Save processed data
        output_path = processor.data_dir / 'processed' / f's{subject:02d}_processed.joblib'
        processor.save_processed_data(features, labels, output_path)
        
if __name__ == '__main__':
    process_deap_dataset() 