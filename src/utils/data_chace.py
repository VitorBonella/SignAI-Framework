import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import List, Optional

class DataCache:
    """Classe para gerenciar cache de dados processados."""
    
    @staticmethod
    def save_features_csv(filepath: str, X: np.ndarray, y: np.ndarray, 
                         folds: np.ndarray, feature_names: List[str]) -> None:
        """
        Salva features em CSV com rótulos e fold index.
        
        Args:
            filepath: Caminho para o arquivo CSV
            X: Array com as features
            y: Array com os rótulos
            folds: Array com os índices dos folds
            feature_names: Lista de nomes das features
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['fold'] = folds
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def load_features_csv(filepath: str) -> pd.DataFrame:
        """
        Carrega CSV de features.
        
        Args:
            filepath: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados carregados
        """
        return pd.read_csv(filepath)
    
    @staticmethod
    def csv_matches_features(filepath: str, feature_names: List[str]) -> bool:
        """
        Verifica se o CSV existente tem as mesmas colunas de features.
        
        Args:
            filepath: Caminho para o arquivo CSV
            feature_names: Lista de nomes de features esperadas
            
        Returns:
            True se as features correspondem, False caso contrário
        """
        if not os.path.exists(filepath):
            return False
        
        df = pd.read_csv(filepath, nrows=1)
        expected_cols = feature_names + ['target', 'fold']
        return all(col in df.columns for col in expected_cols)
    
    @staticmethod
    def get_processed_filename(output_dir: Path, dataset_name: str, 
                             experiment_name: str) -> Path:
        """
        Gera o nome do arquivo processado baseado no dataset e transformações.
        
        Args:
            output_dir: Diretório de saída
            dataset_name: Nome do dataset
            transform_name: Nome da transformação
            
        Returns:
            Caminho completo para o arquivo
        """
        return output_dir / f"{dataset_name}_{experiment_name}_features.csv"