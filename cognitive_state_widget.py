import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveStateWidget:
    def __init__(self, root: Optional[tk.Tk] = None):
        """Initialize the cognitive state monitoring widget.
        
        Args:
            root: Optional Tkinter root window
        """
        self.root = root if root else tk.Tk()
        self.root.title("Cognitive State Monitor")
        
        # Set up the main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize data structures
        self.state_queue = queue.Queue()
        self.history: List[Dict] = []
        self.max_history = 60  # 1 minute of data at 1 reading per second
        
        # Create widgets
        self._create_widgets()
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def _create_widgets(self):
        """Create and arrange the widget components."""
        # Current state display
        self.state_label = ttk.Label(
            self.frame, 
            text="Current State: Unknown",
            font=('Helvetica', 14, 'bold')
        )
        self.state_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Focus level indicator
        self.focus_frame = ttk.LabelFrame(self.frame, text="Focus Level", padding="5")
        self.focus_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.focus_bar = ttk.Progressbar(
            self.focus_frame,
            length=200,
            mode='determinate'
        )
        self.focus_bar.grid(row=0, column=0, padx=5, pady=5)
        
        # Trend graph
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, pady=5)
        
        # Feedback text
        self.feedback_label = ttk.Label(
            self.frame,
            text="Waiting for data...",
            font=('Helvetica', 10)
        )
        self.feedback_label.grid(row=3, column=0, columnspan=2, pady=5)
        
    def update_state(self, state: Dict):
        """Update the widget with new cognitive state data.
        
        Args:
            state: Dictionary containing state information
        """
        self.state_queue.put(state)
        
    def _update_loop(self):
        """Background thread for updating the widget."""
        while self.running:
            try:
                # Get new state from queue
                state = self.state_queue.get(timeout=0.1)
                
                # Update history
                self.history.append(state)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                    
                # Update UI in main thread
                self.root.after(0, self._update_ui, state)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                
    def _update_ui(self, state: Dict):
        """Update the UI components with new state data.
        
        Args:
            state: Dictionary containing state information
        """
        # Update state label
        self.state_label.config(
            text=f"Current State: {state['state'].title()}"
        )
        
        # Update focus bar
        focus_level = state.get('focus_level', 0)
        self.focus_bar['value'] = focus_level
        
        # Update focus bar color
        if focus_level >= 70:
            color = 'green'
        elif focus_level >= 40:
            color = 'yellow'
        else:
            color = 'red'
        self.focus_bar['style'] = f'Horizontal.TProgressbar.{color}'
        
        # Update trend graph
        self._update_graph()
        
        # Update feedback text
        self._update_feedback(state)
        
    def _update_graph(self):
        """Update the trend graph with historical data."""
        self.ax.clear()
        
        if not self.history:
            return
            
        # Extract focus levels
        focus_levels = [h.get('focus_level', 0) for h in self.history]
        times = [h.get('timestamp', i) for i, h in enumerate(self.history)]
        
        # Plot data
        self.ax.plot(times, focus_levels, 'b-')
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Focus Level')
        self.ax.set_title('Focus Trend')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _update_feedback(self, state: Dict):
        """Update the feedback text based on current state.
        
        Args:
            state: Dictionary containing state information
        """
        focus_level = state.get('focus_level', 0)
        current_state = state.get('state', 'unknown')
        
        if current_state == 'engaged' and focus_level >= 70:
            feedback = "You're in deep focus! Great work!"
        elif current_state == 'confused' or focus_level < 40:
            feedback = "Consider taking a short break to refresh."
        elif current_state == 'fatigued':
            feedback = "You might be getting tired. Time for a break?"
        else:
            feedback = "Maintaining good focus. Keep going!"
            
        self.feedback_label.config(text=feedback)
        
    def run(self):
        """Run the widget's main loop."""
        self.root.mainloop()
        
    def stop(self):
        """Stop the widget and clean up."""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join()
        self.root.quit()

# Example usage
if __name__ == '__main__':
    widget = CognitiveStateWidget()
    
    # Simulate some updates
    def simulate_updates():
        import time
        import random
        
        states = ['engaged', 'confused', 'fatigued', 'neutral']
        while True:
            state = {
                'state': random.choice(states),
                'focus_level': random.randint(0, 100),
                'timestamp': datetime.now().isoformat()
            }
            widget.update_state(state)
            time.sleep(1)
            
    # Start simulation in a separate thread
    sim_thread = threading.Thread(target=simulate_updates)
    sim_thread.daemon = True
    sim_thread.start()
    
    # Run the widget
    widget.run() 