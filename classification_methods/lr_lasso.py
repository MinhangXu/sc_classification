# classification/lr_lasso.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from .base import Classifier

class LRLasso(Classifier):
    """Logistic Regression with L1 regularization for feature selection."""
    
    def prepare_data(self, use_factorized=True, factorization_method='X_fa', selected_features=None):
        """
        Prepare the feature matrix (X), target labels (y), and filtered original feature matrix.
        
        Parameters:
        - use_factorized: Whether to use dimensionality-reduced features
        - factorization_method: The key in adata.obsm for the reduced features
        - selected_features: List of feature names to use (if not using factorized)
        
        Returns:
        - X: The processed feature matrix
        - y: The binary target vector
        - feature_names: A list of feature names
        - valid_X_original: The original feature matrix filtered for valid rows
        """
        # Ensure alignment
        assert self.adata.obs.shape[0] == self.adata.X.shape[0], \
            "Mismatch between `adata.obs` and `adata.X`. Check preprocessing."
        
        if use_factorized:
            if factorization_method not in self.adata.obsm:
                raise ValueError(f"{factorization_method} not found in adata.obsm.")
            X = self.adata.obsm[factorization_method]
            feature_names = [f"{factorization_method}_{i+1}" for i in range(X.shape[1])]
        else:
            if selected_features is None:
                selected_features = self.adata.var[self.adata.var['sig'] == True].index
            X = self.adata[:, selected_features].X
            feature_names = selected_features

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        y = (self.adata.obs[self.target_col] == self.target_value).astype(int)

        valid_idx = ~self.adata.obs[self.target_col].isna()
        self.adata = self.adata[valid_idx].copy()
        X = X[valid_idx]
        y = y[valid_idx]
        X_original = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        valid_X_original = X_original

        return X, y, feature_names, valid_X_original
    
    def fit(self, X, y, C=1.0):
        """
        Fit the LR-Lasso classifier.
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        - C: Inverse of regularization strength
        
        Returns:
        - Fitted classifier
        """
        model = LogisticRegression(
            penalty='l1', 
            solver='saga', 
            max_iter=5000,
            random_state=45, 
            class_weight='balanced', 
            C=C
        )
        model.fit(X, y)
        self.model = model
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted classifier.
        
        Parameters:
        - X: Feature matrix
        
        Returns:
        - Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        - X: Feature matrix
        
        Returns:
        - Class probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier.
        
        Parameters:
        - X: Feature matrix
        - y: True labels
        
        Returns:
        - Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred)
        
        # Calculate ROC AUC
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        
        return metrics
    
    def _calculate_metrics(self, y, y_pred):
        """
        Calculate basic classification metrics.
        
        Parameters:
        - y: True labels
        - y_pred: Predicted labels
        
        Returns:
        - Dictionary of metrics
        """
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fn = np.sum((y == 1) & (y_pred == 0))
        overall_accuracy = accuracy_score(y, y_pred)

        if np.sum(y == 1) > 0:
            mal_accuracy = accuracy_score(y[y == 1], y_pred[y == 1])
            mal_precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
            mal_recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
            mal_f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
        else:
            mal_accuracy = mal_precision = mal_recall = mal_f1 = float('nan')

        if np.sum(y == 0) > 0:
            norm_accuracy = accuracy_score(y[y == 0], y_pred[y == 0])
            norm_precision = precision_score(y, y_pred, pos_label=0, zero_division=0)
            norm_recall = recall_score(y, y_pred, pos_label=0, zero_division=0)
            norm_f1 = f1_score(y, y_pred, pos_label=0, zero_division=0)
        else:
            norm_accuracy = norm_precision = norm_recall = norm_f1 = float('nan')

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "overall_accuracy": overall_accuracy,
            "mal_accuracy": mal_accuracy,
            "mal_precision": mal_precision,
            "mal_recall": mal_recall,
            "mal_f1": mal_f1,
            "norm_accuracy": norm_accuracy,
            "norm_precision": norm_precision,
            "norm_recall": norm_recall,
            "norm_f1": norm_f1,
        }
    
    def fit_along_regularization_path(self, X, y, feature_names, alphas=None, metrics_grouping='patient'):
        """
        Fit models along the regularization path and calculate metrics.
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        - feature_names: List of feature names
        - alphas: Array of regularization strengths
        - metrics_grouping: Grouping column name in adata.obs
        
        Returns:
        - Dictionary with coefficients and metrics
        """
        if alphas is None:
            alphas = np.logspace(-4, 5, 20)
        C_values = 1 / alphas

        coefs = []
        feature_elimination_dict = {f: None for f in feature_names}
        group_metrics_path = []

        if metrics_grouping not in ['sample', 'patient']:
            raise ValueError("metrics_grouping must be either 'sample' or 'patient'")

        groups = self.adata.obs[metrics_grouping].unique()

        for i, C in enumerate(C_values):
            print(f"Training model with C={C:.2e} (alpha={alphas[i]:.2e}) ({i+1}/{len(C_values)})")
            self.fit(X, y, C=C)
            coefs.append(self.model.coef_.ravel())

            # Track feature elimination timing
            for idx, coef in enumerate(self.model.coef_.ravel()):
                if coef == 0 and feature_elimination_dict[feature_names[idx]] is None:
                    feature_elimination_dict[feature_names[idx]] = alphas[i]

            # Log metrics for the overall dataset
            y_prob = self.predict_proba(X)[:, 1]
            y_pred = self.predict(X)
            overall_metrics = self._log_metrics_with_probabilities(
                X, y, y_pred, y_prob, alphas, i, group_name="all_samples"
            )
            group_metrics_path.append(overall_metrics)

            # Log metrics for each individual group
            for group in groups:
                group_mask = self.adata.obs[metrics_grouping] == group
                X_group = X[group_mask]
                y_group = y[group_mask]

                print(f"{metrics_grouping.capitalize()}: {group}, Valid Rows: {len(y_group)}")
                if len(y_group) == 0:
                    print(f"Warning: {metrics_grouping.capitalize()} {group} has no valid data after filtering. Skipping.")
                    continue

                y_prob_group = self.predict_proba(X_group)[:, 1]
                y_pred_group = self.predict(X_group)
                group_metrics = self._log_metrics_with_probabilities(
                    X_group, y_group, y_pred_group, y_prob_group, alphas, i, group_name=str(group)
                )
                group_metrics_path.append(group_metrics)

        return {
            'coefs': np.array(coefs).T,
            'feature_elimination_dict': feature_elimination_dict,
            'group_metrics_path': group_metrics_path
        }
    
    def _log_metrics_with_probabilities(self, X, y, y_pred, y_prob, alphas, i, group_name="all_samples"):
        """
        Log metrics for a given dataset (overall or per group).
        
        Parameters:
        - X: Feature matrix
        - y: True labels
        - y_pred: Predicted labels
        - y_prob: Predicted probabilities
        - alphas: Array of alpha values
        - i: Index of the current alpha
        - group_name: Identifier for the group
        
        Returns:
        - Dictionary of metrics
        """
        metrics = self._calculate_metrics(y, y_pred)
        majority_class = pd.Series(y).value_counts().idxmax()
        majority_class_num = pd.Series(y).value_counts().max()
        minority_class_num = pd.Series(y).value_counts().min() if len(pd.Series(y).value_counts()) > 1 else 0

        y_trivial = np.full_like(y, fill_value=majority_class)
        trivial_metrics = self._calculate_metrics(y, y_trivial)

        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        metrics.update({
            "trivial_accuracy": trivial_metrics["overall_accuracy"],
            "majority_num": majority_class_num,
            "minority_num": minority_class_num,
            "roc_auc": roc_auc,
            "group": group_name,
            "alpha": alphas[i],
            "fpr": np.array2string(fpr, separator=','),
            "tpr": np.array2string(tpr, separator=','),
            "thresholds": np.array2string(thresholds, separator=',')
        })
        return metrics