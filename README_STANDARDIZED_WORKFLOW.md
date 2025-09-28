# Standardized Single-Cell Classification Workflow

This document describes the new standardized workflow system for single-cell classification that provides consistent experiment tracking, reproducible results, and easy comparison between different parameter configurations.

## Overview

The standardized workflow addresses the challenges of:
- **Experiment Tracking**: Automatic generation of unique experiment IDs with configuration hashing
- **Result Organization**: Hierarchical directory structure for all intermediate and final results
- **Reproducibility**: Complete configuration tracking and model saving
- **Comparison**: Easy comparison between different experiments (FA vs NMF, different parameters, etc.)
- **Downstream Analysis**: Standardized export format for factor interpretation and projection

## Architecture

### Core Components

1. **Experiment Manager** (`utils/experiment_manager.py`)
   - Manages experiment lifecycle and metadata
   - Generates unique experiment IDs
   - Provides standardized file paths

2. **Preprocessing Pipeline** (`utils/preprocessing.py`)
   - Consistent data filtering and HVG selection
   - Standardized data preprocessing
   - Validation and quality checks

3. **Standardized Pipeline** (`pipeline/standardized_pipeline.py`)
   - Unified pipeline for both FA and NMF
   - Consistent classification workflow
   - Automatic result saving

4. **Experiment Analyzer** (`utils/experiment_analysis.py`)
   - Cross-experiment comparison
   - Performance analysis and visualization
   - Export for downstream analysis

### Directory Structure

```
experiments/
├── {experiment_id}/
│   ├── config.yaml                    # Complete experiment configuration
│   ├── metadata.json                  # Experiment metadata and timestamps
│   ├── preprocessing/
│   │   ├── adata_processed.h5ad       # Final preprocessed data
│   │   ├── hvg_list.pkl              # HVG list
│   │   ├── scaler.pkl                # StandardScaler object
│   │   └── preprocessing_summary.txt # Preprocessing report
│   ├── models/
│   │   ├── {dr_method}_{n_components}/
│   │   │   ├── model.pkl             # DR model object
│   │   │   ├── transformed_data.h5ad # DR-transformed data
│   │   │   └── model_summary.txt     # Model diagnostics
│   │   └── classification/
│   │       ├── {patient_id}/
│   │       │   ├── coefficients.csv  # LR coefficients
│   │       │   ├── metrics.csv       # Classification metrics
│   │       │   └── downsampling_info.json # Downsampling details
│   ├── analysis/
│   │   ├── factor_interpretation/    # GSEA results
│   │   ├── projections/              # Cross-timepoint projections
│   │   └── summary_plots/            # Final visualizations
│   └── logs/                         # Execution logs
```

## Usage

### 1. Running a Standardized Pipeline

#### Command Line Interface

```bash
# Run FA pipeline with 100 components and donor downsampling
python pipeline/standardized_pipeline.py \
    --input /path/to/input_data.h5ad \
    --dr-method fa \
    --n-components 100 \
    --downsampling random \
    --target-fraction 0.5

# Run NMF pipeline with 50 components and no downsampling
python pipeline/standardized_pipeline.py \
    --input /path/to/input_data.h5ad \
    --dr-method nmf \
    --n-components 50 \
    --downsampling none
```

#### Programmatic Interface

```python
from utils.experiment_manager import ExperimentManager, create_standard_config
from pipeline.standardized_pipeline import StandardizedPipeline

# Create experiment manager
experiment_manager = ExperimentManager("experiments")

# Create configuration
config = create_standard_config(
    dr_method='fa',
    n_components=100,
    downsampling_method='random',
    target_donor_fraction=0.5
)

# Run pipeline
pipeline = StandardizedPipeline(experiment_manager)
experiment = pipeline.run_pipeline(config, "/path/to/input_data.h5ad")

print(f"Experiment ID: {experiment.config.experiment_id}")
```

### 2. Comparing Experiments

#### Command Line Interface

```bash
# Compare multiple experiments
python utils/experiment_analysis.py \
    --experiment-ids exp1 exp2 exp3 \
    --generate-report \
    --create-plots \
    --export-results
```

#### Programmatic Interface

```python
from utils.experiment_analysis import ExperimentAnalyzer

# Create analyzer
analyzer = ExperimentAnalyzer(experiment_manager)

# Compare experiments
comparison_df = analyzer.compare_experiments(['exp1', 'exp2', 'exp3'])
print(comparison_df)

# Generate performance comparison
performance_df = analyzer.compare_classification_performance(['exp1', 'exp2', 'exp3'])

# Create comparison plots
analyzer.create_performance_comparison_plots(['exp1', 'exp2', 'exp3'], "comparison_plots")

# Export results for downstream analysis
analyzer.export_results_for_downstream_analysis('exp1', "downstream_analysis")
```

### 3. Accessing Results

```python
# Load an experiment
experiment = experiment_manager.load_experiment("experiment_id")

# Get experiment summary
summary = experiment.get_experiment_summary()
print(summary)

# Access specific files
preprocessed_data_path = experiment.get_path('preprocessed_data')
model_path = experiment.get_path('dr_model', dr_method='fa', n_components=100)
coefficients_path = experiment.get_path('patient_coefficients', patient_id='P01')
```

## Configuration

### Standard Configuration

The `create_standard_config()` function creates a default configuration:

```python
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
        'method': 'fa',  # or 'nmf'
        'n_components': 100,
        'random_state': 42
    },
    'classification': {
        'method': 'lr_lasso',
        'alphas': 'logspace(-4, 5, 20)',
        'cv_folds': 5,
        'random_state': 42
    },
    'downsampling': {
        'method': 'random',  # or 'none'
        'target_donor_fraction': 0.5,
        'min_cells_per_type': 20,
        'cell_type_column': 'predicted.annotation'
    },
    'analysis': {
        'factor_interpretation': True,
        'projection_validation': True,
        'summary_plots': True
    }
}
```

### Custom Configuration

You can create custom configurations by modifying the standard config:

```python
config = create_standard_config()
config.config['dimension_reduction']['n_components'] = 50
config.config['downsampling']['method'] = 'none'
config.config['preprocessing']['n_top_genes'] = 2000
```

## Experiment ID Generation

Experiment IDs are automatically generated based on configuration parameters:

```
{timestamp}_{dr_method}_{n_components}_{downsampling}_{config_hash}
```

Example: `20241201_143022_fa_100_random_a1b2c3d4`

This ensures:
- **Uniqueness**: Identical configurations get the same ID
- **Readability**: Key parameters are visible in the ID
- **Traceability**: Timestamp shows when the experiment was created

## Migration from Old Workflow

### 1. Archive Old Results

```bash
# Create archive of old results
mkdir old_results_archive
mv pipeline/results_* old_results_archive/
mv evaluation/factor_interpretation old_results_archive/
```

### 2. Run New Standardized Pipeline

```bash
# Run FA pipeline (equivalent to your current workflow)
python pipeline/standardized_pipeline.py \
    --input /home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad \
    --dr-method fa \
    --n-components 100 \
    --downsampling random \
    --target-fraction 0.5

# Run NMF pipeline for comparison
python pipeline/standardized_pipeline.py \
    --input /home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad \
    --dr-method nmf \
    --n-components 100 \
    --downsampling random \
    --target-fraction 0.5
```

### 3. Compare Results

```bash
# Get experiment IDs
ls experiments/

# Compare FA vs NMF
python utils/experiment_analysis.py \
    --experiment-ids {fa_exp_id} {nmf_exp_id} \
    --generate-report \
    --create-plots
```

## Downstream Analysis Integration

### Factor Interpretation

The standardized workflow provides easy access to results for factor interpretation:

```python
# Export results for GSEA analysis
analyzer.export_results_for_downstream_analysis('experiment_id', "gsea_analysis")

# The exported directory will contain:
# - all_coefficients.csv: LR coefficients for all patients
# - all_metrics.csv: Classification metrics
# - models/: DR model files for projection
# - experiment_config.json: Complete configuration
```

### Projection Analysis

```python
# Load experiment and model
experiment = experiment_manager.load_experiment('experiment_id')
model_path = experiment.get_path('dr_model', dr_method='fa', n_components=100)

# Load model for projection
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Project to other timepoints
# (Use existing projection scripts with the standardized model paths)
```

## Negative Values in NMF

MultiVI output only non-negative data matrix, the negative value is coming from sklearn standardization right after HVG selection

### Available Methods

1. **`clip`**: Clips negative values to 0
   - Conservative approach
   - May lose information from negative values
   - Example: `[-2, -1, 0, 1, 2]` → `[0, 0, 0, 1, 2]`

2. **`error`**: Raises an error if negative values are found
   - Useful for debugging and validation
   - Helps identify unexpected negative values

### Configuration

Set the method in your experiment configuration:

```python
config = create_standard_config(
    dr_method='nmf',
    n_components=100
)

# Set negative value handling method
config.config['dimension_reduction']['handle_negative_values'] = 'error'
```

## Pipeline Configuration

## Benefits of the New Workflow

1. **Reproducibility**: Every experiment is fully tracked and reproducible
2. **Comparison**: Easy comparison between different methods and parameters
3. **Organization**: Clear, hierarchical structure for all results
4. **Scalability**: Easy to add new dimension reduction methods or analysis types
5. **Integration**: Standardized interfaces for downstream analysis
6. **Documentation**: Automatic generation of experiment summaries and reports

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install scanpy pandas numpy scikit-learn matplotlib seaborn pyyaml jinja2
   ```

2. **Path Issues**: Use absolute paths for input data
   ```bash
   python pipeline/standardized_pipeline.py --input /absolute/path/to/data.h5ad
   ```

3. **Memory Issues**: For large datasets, consider reducing n_components or n_top_genes

### Getting Help

- Check experiment logs in `experiments/{experiment_id}/logs/`
- Review experiment metadata in `experiments/{experiment_id}/metadata.json`
- Use the experiment analyzer to compare successful vs failed experiments

## Future Enhancements

1. **Additional DR Methods**: Easy to add PCA, UMAP, etc.
2. **Advanced Downsampling**: More sophisticated downsampling strategies
3. **Automated Analysis**: Automatic generation of factor interpretation and projections
4. **Web Interface**: Web-based experiment management and visualization
5. **Database Integration**: Store experiment metadata in a database for better querying 