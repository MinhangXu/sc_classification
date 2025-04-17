# evaluation/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_metrics_and_coefficients(metrics_df, coefs_df, sample_name="all_samples", top_n=10):
    """
    Plot classification metrics and coefficient paths.
    
    Parameters:
    - metrics_df: DataFrame containing metrics
    - coefs_df: DataFrame containing coefficients
    - sample_name: Name of the sample to display
    - top_n: Number of top features to display
    
    Returns:
    - Figure object
    """
    # Filter metrics for the given sample
    metrics_results = metrics_df[metrics_df['group'] == sample_name]
    
    if metrics_results.empty:
        print(f"No metrics found for sample '{sample_name}'")
        return None
    
    # Extract alpha values and coefficient data
    alphas = coefs_df.columns.astype(float).values
    coef_results_arr = np.array(coefs_df)
    feature_names = coefs_df.index
    
    # Extract metrics
    overall_acc = metrics_results['overall_accuracy'].values
    mal_accuracy = metrics_results['mal_accuracy'].values
    norm_accuracy = metrics_results['norm_accuracy'].values
    
    # Check if ROC AUC is present
    has_roc = 'roc_auc' in metrics_results.columns
    if has_roc:
        roc_auc = metrics_results['roc_auc'].values
    else:
        roc_auc = None
    
    # Surviving features count
    surviving_features = (coefs_df != 0).sum(axis=0)
    
    # Extract class distribution information
    if 'majority_num' in metrics_results.columns and 'minority_num' in metrics_results.columns:
        majority_num = metrics_results['majority_num'].values[0]
        minority_num = metrics_results['minority_num'].values[0]
    else:
        majority_num = "Unknown"
        minority_num = "Unknown"
    
    # Identify top features based on survival
    non_zero_counts = (coefs_df != 0).sum(axis=1)
    top_features_idx = np.argsort(non_zero_counts)[-top_n:]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # Top plot: Metrics
    log_alphas = np.log10(alphas)
    
    # Accuracy lines
    ax1.plot(log_alphas, overall_acc, 'o-', label="Overall Accuracy", color='skyblue', linewidth=1.5, alpha=0.8)
    ax1.plot(log_alphas, mal_accuracy, '^-', label="Cancer Cell Accuracy", color='darkblue', linewidth=1.5, alpha=0.8)
    ax1.plot(log_alphas, norm_accuracy, 's--', label="Normal Cell Accuracy", color='green', linewidth=1.5, alpha=0.8)
    
    # Trivial accuracy if available
    if "trivial_accuracy" in metrics_results:
        trivial_acc = metrics_results['trivial_accuracy'].values
        ax1.plot(
            log_alphas,
            trivial_acc,
            '--',
            label=f"Trivial (Majority) Accuracy = {trivial_acc[0]:.3f}",
            color='red',
            linewidth=1.5,
            alpha=0.8
        )
    
    # ROC AUC if available
    if roc_auc is not None:
        ax1.plot(
            log_alphas,
            roc_auc,
            'd-',
            label="ROC AUC",
            color='purple',
            linewidth=1.5,
            alpha=0.8
        )
    
    ax1.set_xlabel(r"$\log_{10}(\lambda)$")
    ax1.set_ylabel("Accuracy / AUC")
    ax1.grid(True)
    
    # Secondary y-axis for surviving features
    ax1_2 = ax1.twinx()
    ax1_2.plot(log_alphas,
               surviving_features / len(feature_names) * 100,
               's-',
               color='orange',
               label="Surviving Features (%)",
               alpha=0.8)
    ax1_2.set_ylabel("Surviving Features (%)")
    ax1_2.set_ylim([0, 100])
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_2.get_legend_handles_labels()
    
    # Add class distribution information
    extra_lines = [plt.Line2D([0], [0], color="none")] * 2
    extra_labels = [
        f"Majority (normal) Size: {majority_num}",
        f"Minority (cancer) Size: {minority_num}"
    ]
    
    ax1.legend(
        extra_lines + lines_1 + lines_2,
        extra_labels + labels_1 + labels_2,
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        fontsize='small',
        frameon=False
    )
    
    ax1.set_title(f"Model Accuracy and Metrics\nSample: {sample_name}")
    
    # Bottom plot: Coefficient paths
    for idx in top_features_idx:
        ax2.plot(alphas, coef_results_arr[idx],
                 label=feature_names[idx],
                 alpha=0.8)
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r"$\lambda$ (regularization strength)")
    ax2.set_ylabel("Coefficient Value")
    ax2.set_title("Logistic Regression Coefficient Paths for Top Features")
    ax2.axhline(0, color='black', linestyle='--', lw=1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize='small')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(metrics_df, sample_name, figsize=(15, 10)):
    """
    Plot confusion matrices for a specific sample across alphas.
    
    Parameters:
    - metrics_df: DataFrame containing metrics with TP, FP, TN, FN values
    - sample_name: Name of the sample to display
    - figsize: Size of the figure
    
    Returns:
    - Figure object
    """
    # Filter for the given sample
    sample_metrics = metrics_df[metrics_df['group'] == sample_name]
    
    if sample_metrics.empty:
        print(f"No metrics found for sample '{sample_name}'")
        return None
    
    # Extract confusion matrix components
    alphas = sample_metrics['alpha'].values
    tp = sample_metrics['tp'].values
    fp = sample_metrics['fp'].values
    tn = sample_metrics['tn'].values
    fn = sample_metrics['fn'].values
    
    # Determine grid layout
    n_alphas = len(alphas)
    n_cols = min(n_alphas, 5)
    n_rows = (n_alphas + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each confusion matrix
    import seaborn as sns
    
    for i, alpha in enumerate(alphas):
        if i < len(axes):
            # Create confusion matrix
            cm = np.array([[tp[i], fp[i]], [fn[i], tn[i]]])
            
            # Plot as heatmap
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
            axes[i].set_title(f"Alpha: {alpha:.2e}", fontsize=10)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
            axes[i].set_xticklabels(["Cancer", "Normal"], fontsize=8)
            axes[i].set_yticklabels(["Cancer", "Normal"], fontsize=8)
    
    # Hide unused axes
    for i in range(n_alphas, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f"Confusion Matrices Across Alphas\nSample: {sample_name}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def plot_roc_curves(metrics_df, sample_name, figsize=(12, 10)):
    """
    Plot ROC curves for a specific sample across alphas.
    
    Parameters:
    - metrics_df: DataFrame containing metrics with FPR, TPR values
    - sample_name: Name of the sample to display
    - figsize: Size of the figure
    
    Returns:
    - Figure object
    """
    # Filter for the given sample
    sample_metrics = metrics_df[metrics_df['group'] == sample_name]
    
    if sample_metrics.empty:
        print(f"No metrics found for sample '{sample_name}'")
        return None
    
    # Check if FPR and TPR columns exist
    if 'fpr' not in sample_metrics.columns or 'tpr' not in sample_metrics.columns:
        print(f"ROC curve data not found for sample '{sample_name}'")
        return None
    
    # Parse string arrays to numpy arrays
    sample_metrics['fpr'] = sample_metrics['fpr'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x
    )
    sample_metrics['tpr'] = sample_metrics['tpr'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x
    )
    
    # Filter samples with minority class
    valid_metrics = sample_metrics[sample_metrics['minority_num'] > 0]
    if valid_metrics.empty:
        print(f"No valid ROC data for sample '{sample_name}'")
        return None
    
    # Sort by alpha
    valid_metrics = valid_metrics.sort_values('alpha').reset_index(drop=True)
    alphas = valid_metrics['alpha'].values
    
    # Create subplots
    n_alphas = len(alphas)
    n_cols = min(5, n_alphas)
    n_rows = (n_alphas + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot ROC curves
    for i, idx in enumerate(valid_metrics.index):
        if i < len(axes):
            fpr = valid_metrics.loc[idx, 'fpr']
            tpr = valid_metrics.loc[idx, 'tpr']
            alpha = valid_metrics.loc[idx, 'alpha']
            roc_auc = valid_metrics.loc[idx, 'roc_auc']
            
            # Plot ROC curve
            axes[i].plot(fpr, tpr, 
                      label=f'AUC = {roc_auc:.3f}', 
                      color='b', 
                      linewidth=2)
            axes[i].plot([0, 1], [0, 1], 'k--')  # Random classifier line
            axes[i].set_title(f"Alpha = {alpha:.2e}", fontsize=10)
            axes[i].set_xlabel("False Positive Rate")
            axes[i].set_ylabel("True Positive Rate")
            axes[i].legend(loc="lower right", fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_alphas, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f"ROC Curves Across Alphas\nSample: {sample_name}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig