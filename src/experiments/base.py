# vibclassifier/experiments/base.py
from abc import ABC, abstractmethod
import json
from typing import Optional, Dict, Any
from vibdata.raw.base import RawVibrationDataset
from vibdata.deep.signal.transforms import Transform

class Experiment(ABC):
    """Classe base abstrata para todos os experimentos de classificação de vibração."""
    
    def __init__(
        self,
        name: str,
        description: str,
        dataset: Optional[RawVibrationDataset] = None,
        data_transform: Optional[Transform] = None,
        feature_selector = None,
        model = None
    ):
        """
        Inicializa o experimento.
        
        Args:
            name: Nome identificador do experimento
            description: Descrição detalhada do experimento
            dataset: Conjunto de dados de vibração
            data_transform: Transformação a ser aplicada nos dados brutos
            data_division_method: Método de divisão dos dados (e.g., 'kfold', 'holdout')
            data_division_params: Parâmetros para o método de divisão
            feature_selector: Seletor de features (para experimentos com extração)
            model: Modelo de machine learning/deep learning
        """
        self.name = name
        self.description = description
        self.dataset = dataset
        self.data_transform = data_transform
        self.feature_selector = feature_selector
        self.model = model
        
        # Resultados serão armazenados aqui
        self.results = {}
    
    @abstractmethod
    def prepare_data(self):
        """Prepara os dados para o experimento."""
        pass
    
    @abstractmethod
    def run(self):
        """Executa o experimento completo."""
        pass
    
    def save_results(self, filepath: str):
        """Salva os resultados do experimento."""
        # Implementação básica - pode ser extendida
        with open(filepath, 'w') as f:
            json.dump(self.results, f)
    
    def load_results(self, filepath: str):
        """Carrega resultados de um experimento anterior."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    def __str__(self):
        return f"Experiment: {self.name}\nDescription: {self.description}"