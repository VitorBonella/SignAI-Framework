import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pathlib import Path

from vibdata.raw.base import RawVibrationDataset
from vibdata.deep.signal.transforms import Transform
from signalai.utils.logging import setup_logger

class BaseExperiment(ABC):
    """Abstract base class for all vibration classification experiments."""
    
    def __init__(
        self,
        name: str,
        description: str,
        dataset: Optional[Any] = None, # Using Any to avoid strict vibdata dependency if needed
        data_transform: Optional[Transform] = None,
        feature_selector = None,
        model = None,
        output_dir: Union[str, Path] = "results",
        start_time: Optional[str] = None
    ):
        """
        Initializes the experiment.
        
        Args:
            name: Identifier name for the experiment
            description: Detailed description
            dataset: Vibration dataset
            data_transform: Transformation to apply to raw data
            feature_selector: Feature selector
            model: Machine learning / deep learning model
            output_dir: Base directory for results
            start_time: Optional pre-defined start time (YYYYMMDD_HHMMSS)
        """
        self.name = name
        self.description = description
        self.dataset = dataset
        self.data_transform = data_transform
        self.feature_selector = feature_selector
        self.model = model
        self.output_dir = Path(output_dir)
        
        # Results storage
        self.results = {}
        self.start_time = start_time
        self.run_dir = None
        self.logger = None

        self._setup_run_directory()

    def _setup_run_directory(self):
        """Creates a unique directory for this experiment run and initializes logging."""
        if self.start_time is None:
            self.start_time = time.strftime("%Y%m%d_%H%M%S")
        
        dir_name = f"results_{self.name}_{self.start_time}"
        self.run_dir = self.output_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.run_dir / "experiment.log"
        self.logger = setup_logger(str(log_file))
        print(f"Logging initialized at {log_file}")
    
    @abstractmethod
    def prepare_data(self):
        """Prepare data for the experiment."""
        pass
    
    @abstractmethod
    def run(self):
        """Execute the complete experiment."""
        pass
    
    def save_results(self, filepath: str):
        """Save experiment results."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f)
    
    def load_results(self, filepath: str):
        """Load results from a previous experiment."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    def __str__(self):
        return f"Experiment: {self.name}\nDescription: {self.description}"
