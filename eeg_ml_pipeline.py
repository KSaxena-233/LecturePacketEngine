import random
import time
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDataBuffer:
    def __init__(self, window_size: int = 20):
        """Initialize EEG data buffer with a rolling window.
        
        Args:
            window_size: Number of seconds to keep in buffer
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
    def add_reading(self, reading: Dict):
        """Add a new EEG reading to the buffer."""
        with self.lock:
            self.buffer.append(reading)
            
    def get_window(self) -> List[Dict]:
        """Get all readings in the current window."""
        with self.lock:
            return list(self.buffer)

class EEGSimulator:
    def __init__(self, buffer: EEGDataBuffer):
        """Initialize EEG simulator.
        
        Args:
            buffer: EEGDataBuffer instance to store readings
        """
        self.buffer = buffer
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def simulate_eeg_stream(self):
        """Generate simulated EEG readings."""
        while self.running:
            # Generate random attention and meditation values
            reading = {
                "timestamp": datetime.now().isoformat(),
                "attention": random.randint(0, 100),
                "meditation": random.randint(0, 100)
            }
            
            # Add to buffer
            self.buffer.add_reading(reading)
            
            # Log the reading
            logger.debug(f"Generated EEG reading: {reading}")
            
            # Sleep for random interval between 2-5 seconds
            time.sleep(random.uniform(2, 5))
            
    def start(self):
        """Start the EEG simulation in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self.simulate_eeg_stream)
        self.thread.daemon = True
        self.thread.start()
        logger.info("EEG simulation started")
        
    def stop(self):
        """Stop the EEG simulation."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("EEG simulation stopped")

class EEGStateAnalyzer:
    def __init__(self, buffer: EEGDataBuffer):
        """Initialize EEG state analyzer.
        
        Args:
            buffer: EEGDataBuffer instance to analyze
        """
        self.buffer = buffer
        
    def analyze_state(self) -> str:
        """Analyze current EEG state and return cognitive state label."""
        readings = self.buffer.get_window()
        if not readings:
            return "unknown"
            
        # Calculate average attention and meditation
        avg_attention = np.mean([r["attention"] for r in readings])
        avg_meditation = np.mean([r["meditation"] for r in readings])
        
        # Simple rule-based state classification
        if avg_attention < 30:
            return "fatigued"
        elif avg_attention > 70 and avg_meditation > 60:
            return "engaged"
        elif avg_attention < 50:
            return "confused"
        else:
            return "neutral"

# Global state context
class GlobalState:
    def __init__(self):
        self.buffer = EEGDataBuffer()
        self.simulator = EEGSimulator(self.buffer)
        self.analyzer = EEGStateAnalyzer(self.buffer)
        self.current_state = "unknown"
        self.state_lock = threading.Lock()
        
    def update_state(self):
        """Update the current cognitive state."""
        with self.state_lock:
            self.current_state = self.analyzer.analyze_state()
            
    def get_state(self) -> str:
        """Get the current cognitive state."""
        with self.state_lock:
            return self.current_state

# Create global instance
global_state = GlobalState()

def start_eeg_monitoring():
    """Start EEG monitoring system."""
    global_state.simulator.start()
    
    # Start state update loop
    def update_loop():
        while True:
            global_state.update_state()
            time.sleep(5)  # Update state every 5 seconds
            
    update_thread = threading.Thread(target=update_loop)
    update_thread.daemon = True
    update_thread.start()
    
def stop_eeg_monitoring():
    """Stop EEG monitoring system."""
    global_state.simulator.stop()

def get_current_state() -> str:
    """Get the current cognitive state."""
    return global_state.get_state() 