#!/usr/bin/env python
"""
Experiment Management System for Single-Cell Classification Pipeline

This module provides a standardized framework for:
1. Experiment configuration and tracking
2. Hierarchical result organization
3. Reproducible pipeline execution
4. Cross-experiment comparison
"""

import os
import json
import yaml
import hashlib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import scanpy as sc
from sklearn.preprocessing import StandardScaler


class ExperimentConfig:
    """Configuration class for experiment parameters."""
    
    def __init__(self, config_dict: Dict[str, Any], experiment_id: Optional[str] = None):
        self.config = config_dict
        # If experiment_id is provided, use it (for loading existing experiments)
        # Otherwise, generate a new one (for creating new experiments)
        if experiment_id is not None:
            self.experiment_id = experiment_id
        else:
            self.experiment_id = self._generate_experiment_id()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on configuration."""
        # Create a deterministic string representation
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Include key parameters in ID for readability
        dr_method = self.config.get('dimension_reduction', {}).get('method', 'unknown')
        n_components = self.config.get('dimension_reduction', {}).get('n_components', 'unknown')
        downsampling = self.config.get('downsampling', {}).get('method', 'none')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{dr_method}_{n_components}_{downsampling}_{config_hash}"
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


class ExperimentManager:
    """Main experiment management class."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_experiment(self, config: ExperimentConfig) -> 'Experiment':
        """Create a new experiment instance."""
        return Experiment(self.base_dir, config)
    
    def load_experiment(self, experiment_id: str) -> 'Experiment':
        """Load an existing experiment."""
        experiment_dir = self.base_dir / experiment_id
        print(f"Loading experiment from: {experiment_dir}")
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = ExperimentConfig(config_dict, experiment_id=experiment_id)
        return Experiment(self.base_dir, config)
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        experiments = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "config.yaml").exists():
                experiments.append(item.name)
        return sorted(experiments)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                exp = self.load_experiment(exp_id)
                metadata = exp.load_metadata()
                
                # Extract key metrics
                comparison_data.append({
                    'experiment_id': exp_id,
                    'dr_method': exp.config.get('dimension_reduction.method'),
                    'n_components': exp.config.get('dimension_reduction.n_components'),
                    'downsampling': exp.config.get('downsampling.method'),
                    'creation_date': metadata.get('creation_date'),
                    'status': metadata.get('status', 'unknown')
                })
            except Exception as e:
                print(f"Error loading experiment {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)


class Experiment:
    """Individual experiment instance."""
    
    def __init__(self, base_dir: Path, config: ExperimentConfig):
        self.base_dir = base_dir
        self.config = config
        self.experiment_dir = base_dir / config.experiment_id
        
        # Create directory structure
        self._create_directory_structure()
        
        # Save configuration
        self._save_config()
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _create_directory_structure(self):
        """Create the standardized directory structure."""
        directories = [
            "preprocessing",
            "models",
            "models/classification",
            "analysis",
            "analysis/factor_interpretation",
            "analysis/projections", 
            "analysis/summary_plots",
            "logs"
        ]
        
        for dir_name in directories:
            (self.experiment_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.config, f, default_flow_style=False)
    
    def _initialize_metadata(self):
        """Initialize experiment metadata."""
        metadata = {
            'experiment_id': self.config.experiment_id,
            'creation_date': datetime.now().isoformat(),
            'status': 'initialized',
            'stages_completed': []
        }
        
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_metadata(self, updates: Dict[str, Any]):
        """Update experiment metadata."""
        metadata_path = self.experiment_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata.update(updates)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load experiment metadata."""
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_path(self, path_type: str, **kwargs) -> Path:
        """Get standardized paths for different experiment components."""
        base_paths = {
            'preprocessed_data': self.experiment_dir / "preprocessing" / "adata_processed.h5ad",
            'hvg_list': self.experiment_dir / "preprocessing" / "hvg_list.pkl",
            'scaler': self.experiment_dir / "preprocessing" / "scaler.pkl",
            'dr_model': self.experiment_dir / "models" / f"{kwargs.get('dr_method', 'unknown')}_{kwargs.get('n_components', 'unknown')}" / "model.pkl",
            'transformed_data': self.experiment_dir / "models" / f"{kwargs.get('dr_method', 'unknown')}_{kwargs.get('n_components', 'unknown')}" / "transformed_data.h5ad",
            'patient_coefficients': self.experiment_dir / "models" / "classification" / f"{kwargs.get('patient_id', 'unknown')}" / "coefficients.csv",
            'patient_metrics': self.experiment_dir / "models" / "classification" / f"{kwargs.get('patient_id', 'unknown')}" / "metrics.csv",
            'factor_interpretation': self.experiment_dir / "analysis" / "factor_interpretation",
            'projections': self.experiment_dir / "analysis" / "projections",
            'summary_plots': self.experiment_dir / "analysis" / "summary_plots"
        }
        
        if path_type not in base_paths:
            raise ValueError(f"Unknown path type: {path_type}")
        
        return base_paths[path_type]
    
    def save_preprocessing_results(self, adata: sc.AnnData, hvg_list: List[str], 
                                 scaler: StandardScaler, summary: str):
        """Save preprocessing results."""
        # Save processed AnnData
        adata.write_h5ad(self.get_path('preprocessed_data'))
        
        # Save HVG list
        with open(self.get_path('hvg_list'), 'wb') as f:
            pickle.dump(hvg_list, f)
        
        # Save scaler
        with open(self.get_path('scaler'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save summary
        summary_path = self.experiment_dir / "preprocessing" / "preprocessing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Update metadata
        self.update_metadata({
            'status': 'preprocessing_complete',
            'stages_completed': ['preprocessing'],
            'preprocessing': {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'n_hvgs': len(hvg_list),
                'completion_date': datetime.now().isoformat()
            }
        })
    
    def save_dr_results(self, model, transformed_adata: sc.AnnData, 
                       dr_method: str, n_components: int, summary: str):
        """Save dimension reduction results."""
        model_dir = self.experiment_dir / "models" / f"{dr_method}_{n_components}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save transformed data
        transformed_adata.write_h5ad(model_dir / "transformed_data.h5ad")
        
        # Save summary
        with open(model_dir / "model_summary.txt", 'w') as f:
            f.write(summary)
        
        # Update metadata
        current_metadata = self.load_metadata()
        stages = current_metadata.get('stages_completed', [])
        if 'dr' not in stages:
            stages.append('dr')
        
        self.update_metadata({
            'status': 'dr_complete',
            'stages_completed': stages,
            'dr': {
                'method': dr_method,
                'n_components': n_components,
                'completion_date': datetime.now().isoformat()
            }
        })
    
    def save_classification_results(self, patient_id: str, coefficients: pd.Series, 
                                  metrics, downsampling_info: Dict[str, Any]):   
        """Save classification results for a patient."""
        patient_dir = self.experiment_dir / "models" / "classification" / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save coefficients
        coefficients.to_csv(patient_dir / "coefficients.csv")
        
        # Save metrics
        if not isinstance(metrics, pd.DataFrame):
            metrics_df = pd.DataFrame(metrics)
        else:
            metrics_df = metrics
        metrics_df.to_csv(patient_dir / "metrics.csv", index=False)
        
        # Save downsampling info
        with open(patient_dir / "downsampling_info.json", 'w') as f:
            json.dump(downsampling_info, f, indent=2)
        
        # Update metadata
        current_metadata = self.load_metadata()
        stages = current_metadata.get('stages_completed', [])
        if 'classification' not in stages:
            stages.append('classification')
        
        patients_completed = current_metadata.get('patients_completed', [])
        if patient_id not in patients_completed:
            patients_completed.append(patient_id)
        
        self.update_metadata({
            'status': 'classification_in_progress',
            'stages_completed': stages,
            'patients_completed': patients_completed, # The updated list is saved here
            'classification': {
                'last_patient': patient_id,
                'last_completion_date': datetime.now().isoformat()
            }
        })
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the experiment."""
        metadata = self.load_metadata()
        config = self.config.config
        
        summary = {
            'experiment_id': self.config.experiment_id,
            'creation_date': metadata.get('creation_date'),
            'status': metadata.get('status'),
            'stages_completed': metadata.get('stages_completed', []),
            'configuration': {
                'dimension_reduction': config.get('dimension_reduction', {}),
                'classification': config.get('classification', {}),
                'downsampling': config.get('downsampling', {}),
                'preprocessing': config.get('preprocessing', {})
            }
        }
        
        # Add stage-specific information
        if 'preprocessing' in metadata:
            summary['preprocessing_info'] = metadata['preprocessing']
        
        if 'dr' in metadata:
            summary['dr_info'] = metadata['dr']
        
        if 'patients_completed' in metadata:
            summary['patients_completed'] = metadata['patients_completed']
        
        return summary


def create_standard_config(dr_method: str = 'fa', n_components: int = 100,
                          downsampling_method: str = 'random', 
                          target_donor_fraction: float = 0.5) -> ExperimentConfig:
    """Create a standard experiment configuration."""
    
    config = {
        'preprocessing': {
            'n_top_genes': 3000,
            'standardize': True,
            'timepoint_filter': 'MRD',
            'target_column': 'CN.label',
            'positive_class': 'cancer',
            'negative_class': 'normal'
        },
        'dimension_reduction': {
            'method': dr_method,
            'n_components': n_components,
            'random_state': 42,
            'svd_method': 'lapack',  # For FA
            'beta_loss': 'kullback-leibler',  # For NMF
            'handle_negative_values': 'error'
        },
        'classification': {
            'method': 'lr_lasso',
            'alphas': 'logspace(-4, 5, 20)',
            'cv_folds': 5,
            'random_state': 42
        },
        'downsampling': {
            'method': downsampling_method,
            'target_donor_fraction': target_donor_fraction,
            'target_donor_recipient_ratio_threshold': 10.0,
            'min_cells_per_type': 20,
            'cell_type_column': 'predicted.annotation',
            'donor_recipient_column': 'source'
        },
        'analysis': {
            'factor_interpretation': True,
            'projection_validation': True,
            'summary_plots': True
        }
    }
    
    return ExperimentConfig(config) 