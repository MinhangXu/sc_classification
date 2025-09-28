# Downstream Analysis Guide for Standardized Pipeline

This guide explains how to perform the three main types of downstream analysis on your standardized pipeline experiments:

1. **LR-Lasso Result Analysis**
2. **Factor Interpretation Analysis** 
3. **Projection Validation Analysis**

## Quick Start

### Automated Analysis (Recommended for routine tasks)

```bash
# Run all analyses on your experiment
python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type all \
    --validation-data /path/to/full_data.h5ad

# Run specific analysis
python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type lr_lasso
```

### Interactive Analysis (Recommended for exploration)

```bash
# Start Jupyter and open the template
jupyter notebook analysis_notebook_template.ipynb
```

## 1. LR-Lasso Result Analysis

### What it does:
- Creates two-panel plots showing metrics vs regularization strength
- Shows coefficient sparsity patterns
- Generates factor-specific plots (gene communalities for FA, component importance for NMF)

### Automated usage:
```bash
python run_analysis.py \
    --experiment-id YOUR_EXPERIMENT_ID \
    --analysis-type lr_lasso \
    --output-dir lr_lasso_results
```

### Interactive usage:
1. Open `analysis_notebook_template.ipynb`
2. Set your experiment ID
3. Run the "LR-Lasso Result Analysis" section
4. Modify patient selection and visualization parameters as needed

### Output files:
- `patient_{patient_id}_lr_lasso_analysis.png`: Two-panel plots for each patient
- `gene_communalities_distribution.png`: Gene communalities (FA only)
- `factor_variance_explained.png`: Factor variance explained (FA only)
- `nmf_component_importance.png`: Component importance (NMF only)

## 2. Factor Interpretation Analysis

### What it does:
- **Overall interpretation**: Maps LR coefficients with factor loadings
- **Loading-based interpretation**: Analyzes factor loadings to understand biological meaning
- Creates heatmaps and gene lists for top factors

### Automated usage:
```bash
# Run both interpretation methods
python run_analysis.py \
    --experiment-id YOUR_EXPERIMENT_ID \
    --analysis-type factor_interpretation \
    --factor-interpretation-method both

# Run only loading-based interpretation
python run_analysis.py \
    --experiment-id YOUR_EXPERIMENT_ID \
    --analysis-type factor_interpretation \
    --factor-interpretation-method loading
```

### Interactive usage:
1. Open `analysis_notebook_template.ipynb`
2. Set your experiment ID
3. Run the "Factor Interpretation Analysis" section
4. Modify factor selection and gene analysis parameters

### Output files:
- `patient_{patient_id}_factor_importance.png`: Factor importance plots
- `factor_loading_heatmap.png`: Heatmap of factor loadings
- Gene lists for top factors

## 3. Projection Validation Analysis

### What it does:
- Projects selected factors to other timepoints (preSCT, Relapse)
- Validates factor selection across different biological conditions
- Creates UMAP and scatter plots for projected data

### Requirements:
- Full AnnData with all timepoints (not just MRD)
- Path to validation data file

### Automated usage:
```bash
python run_analysis.py \
    --experiment-id YOUR_EXPERIMENT_ID \
    --analysis-type projection \
    --validation-data /path/to/full_data.h5ad \
    --output-dir projection_results
```

### Interactive usage:
1. Open `analysis_notebook_template.ipynb`
2. Set your experiment ID and validation data path
3. Run the "Projection Validation Analysis" section
4. Modify patient and timepoint selection

### Output files:
- `patient_{patient_id}_{timepoint}_umap.png`: UMAP plots for projected data
- `patient_{patient_id}_{timepoint}_scatter.png`: Scatter plots for 2D projections

## Advanced Usage

### Comparing Multiple Experiments

```python
from utils.experiment_analysis import ExperimentAnalyzer
from utils.experiment_manager import ExperimentManager

# Initialize
experiment_manager = ExperimentManager("experiments")
analyzer = ExperimentAnalyzer(experiment_manager)

# Compare experiments
experiment_ids = ["exp1", "exp2", "exp3"]
comparison_df = analyzer.compare_experiments(experiment_ids)
performance_df = analyzer.compare_classification_performance(experiment_ids)

print(comparison_df)
print(performance_df)
```

### Custom Analysis

```python
# Extract results for custom analysis
results = analyzer.extract_classification_results("YOUR_EXPERIMENT_ID")
metrics_df = results['metrics']
coef_df = results['coefficients']

# Custom plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Compare AUC across patients
plt.figure(figsize=(12, 6))
sns.boxplot(data=metrics_df, x='patient_id', y='auc')
plt.title('AUC Comparison Across Patients')
plt.xticks(rotation=45)
plt.show()
```

### Integration with Existing Notebooks

The automated analysis system is designed to complement your existing notebooks:

1. **Use automated scripts** for routine analysis and batch processing
2. **Use interactive notebooks** for exploration and custom visualizations
3. **Export results** from automated scripts and import into notebooks for further analysis

## File Structure

After running analysis, you'll have:

```
analysis_results/
├── lr_lasso_analysis/
│   ├── patient_P01_lr_lasso_analysis.png
│   ├── patient_P02_lr_lasso_analysis.png
│   ├── gene_communalities_distribution.png
│   └── factor_variance_explained.png
├── factor_interpretation/
│   ├── patient_P01_factor_importance.png
│   ├── factor_loading_heatmap.png
│   └── top_genes_factor_1.txt
└── projection_validation/
    ├── patient_P01_preSCT_umap.png
    ├── patient_P01_Relapse_scatter.png
    └── projection_summary.txt
```

## Troubleshooting

### Common Issues

1. **Experiment not found**
   ```bash
   # Check available experiments
   ls experiments/
   ```

2. **Missing validation data**
   ```bash
   # Ensure path is correct
   ls /path/to/your/validation_data.h5ad
   ```

3. **No classification results**
   - Check that the experiment completed successfully
   - Verify that patients were processed

4. **Memory issues**
   - Reduce number of factors/components
   - Use smaller validation datasets
   - Process patients individually

### Getting Help

- Check experiment logs: `experiments/{experiment_id}/logs/`
- Review experiment metadata: `experiments/{experiment_id}/metadata.json`
- Use the interactive notebook for debugging

## Best Practices

1. **Start with automated analysis** to get a quick overview
2. **Use interactive notebooks** for detailed exploration
3. **Save custom visualizations** for publication
4. **Document your analysis parameters** for reproducibility
5. **Compare multiple experiments** to understand method differences

## Integration with Existing Workflow

The downstream analysis system integrates seamlessly with your existing workflow:

- **Input**: Standardized pipeline experiment results
- **Processing**: Automated scripts + interactive notebooks
- **Output**: Publication-ready figures and data for further analysis

This provides the best of both worlds: automation for routine tasks and flexibility for custom analysis. 