import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGMLClassifier:
    def __init__(self, model_path: str = 'models/eeg_classifier.joblib'):
        """Initialize the ML-based EEG classifier.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Define feature columns based on EEG processing
        self.feature_columns = []
        # Frequency band features
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            self.feature_columns.extend([f'{band}_power_{i}' for i in range(32)])  # 32 channels
        # Statistical features
        self.feature_columns.extend([
            f'mean_{i}' for i in range(32)
        ] + [
            f'std_{i}' for i in range(32)
        ] + [
            f'entropy_{i}' for i in range(32)
        ])
        
    def extract_features(self, readings: List[Dict]) -> np.ndarray:
        """Extract features from EEG readings.
        
        Args:
            readings: List of EEG readings with attention and meditation values
            
        Returns:
            Feature vector for classification
        """
        if not readings:
            return np.zeros(len(self.feature_columns))
            
        # Convert to numpy array
        data = np.array([[r['attention'], r['meditation']] for r in readings])
        
        # Calculate features
        features = []
        
        # Calculate power in each frequency band
        for band_name, (low, high) in {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }.items():
            # Bandpass filter
            b, a = signal.butter(4, [low/50, high/50], btype='band')
            filtered = signal.filtfilt(b, a, data)
            
            # Calculate power
            power = np.mean(filtered ** 2, axis=0)
            features.extend(power)
            
        # Add statistical features
        features.extend([
            np.mean(data, axis=0),  # Mean
            np.std(data, axis=0),   # Standard deviation
            signal.entropy(data, axis=0)  # Spectral entropy
        ])
        
        return np.array(features)
        
    def train(self, training_data: List[Dict], labels: List[str]):
        """Train the classifier on labeled data.
        
        Args:
            training_data: List of EEG reading windows
            labels: List of corresponding state labels
        """
        # Extract features
        X = np.array([self.extract_features(window) for window in training_data])
        y = np.array(labels)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased for better performance
            max_depth=15,      # Increased for more complex patterns
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full training set
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = self.model.score(X_test, y_test)
        logger.info(f"Test set accuracy: {test_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 most important features:")
        logger.info(feature_importance.head(10))
        
        # Save model
        self.save_model()
        
    def predict(self, readings: List[Dict]) -> str:
        """Predict cognitive state from EEG readings.
        
        Args:
            readings: List of recent EEG readings
            
        Returns:
            Predicted cognitive state
        """
        if not self.model:
            self.load_model()
            
        if not self.model:
            return "unknown"
            
        # Extract features
        features = self.extract_features(readings)
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        return self.model.predict(features)[0]
        
    def save_model(self):
        """Save the trained model and scaler."""
        if not self.model:
            return
            
        # Create models directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.scaler = StandardScaler()

class EEGDataLogger:
    def __init__(self, log_path: str = 'data/eeg_logs.csv'):
        """Initialize EEG data logger.
        
        Args:
            log_path: Path to save EEG logs
        """
        self.log_path = log_path
        self.columns = ['timestamp', 'attention', 'meditation', 'state', 
                       'content_type', 'user_feedback']
        
        # Create log file if it doesn't exist
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            pd.DataFrame(columns=self.columns).to_csv(log_path, index=False)
            
    def log_reading(self, reading: Dict, state: str, content_type: str, 
                   user_feedback: float = None):
        """Log an EEG reading with associated metadata.
        
        Args:
            reading: EEG reading dictionary
            state: Current cognitive state
            content_type: Type of content being presented
            user_feedback: Optional user feedback score
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'attention': reading['attention'],
            'meditation': reading['meditation'],
            'state': state,
            'content_type': content_type,
            'user_feedback': user_feedback
        }
        
        # Append to CSV
        pd.DataFrame([log_data]).to_csv(
            self.log_path, 
            mode='a', 
            header=False, 
            index=False
        )
        
    def get_training_data(self) -> Tuple[List[Dict], List[str]]:
        """Get labeled training data from logs.
        
        Returns:
            Tuple of (training_data, labels)
        """
        try:
            # Read logs
            logs = pd.read_csv(self.log_path)
            
            # Group by state and create windows
            training_data = []
            labels = []
            
            for state in logs['state'].unique():
                state_logs = logs[logs['state'] == state]
                
                # Create windows of 10 readings
                for i in range(0, len(state_logs), 10):
                    window = state_logs.iloc[i:i+10]
                    if len(window) == 10:  # Only use complete windows
                        readings = window[['attention', 'meditation']].to_dict('records')
                        training_data.append(readings)
                        labels.append(state)
                        
            return training_data, labels
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return [], [] 