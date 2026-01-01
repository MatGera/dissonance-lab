import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class LogisticRegressionProbe:
    """Linear probe for binary classification on activations."""
    
    def __init__(self, max_iter: int = 1000):
        """Initialize probe.
        
        Args:
            max_iter: Maximum iterations for training
        """
        self.lr = LogisticRegression(max_iter=max_iter, random_state=42)
        self.is_fitted = False
    
    def fit(self, acts: torch.Tensor, labels: torch.Tensor) -> "LogisticRegressionProbe":
        """Fit probe on activations.
        
        Args:
            acts: Activation tensor (N, D)
            labels: Binary labels (N,)
        
        Returns:
            Self for chaining
        """
        X = acts.cpu().numpy()
        y = labels.cpu().numpy()
        self.lr.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, acts: torch.Tensor) -> torch.Tensor:
        """Predict labels.
        
        Args:
            acts: Activation tensor (N, D)
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")
        
        X = acts.cpu().numpy()
        return torch.tensor(self.lr.predict(X))
    
    def predict_proba(self, acts: torch.Tensor) -> torch.Tensor:
        """Predict probabilities.
        
        Args:
            acts: Activation tensor (N, D)
        
        Returns:
            Predicted probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")
        
        X = acts.cpu().numpy()
        probs = self.lr.predict_proba(X)
        return torch.tensor(probs[:, 1])
    
    def score(self, acts: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute accuracy and AUC.
        
        Args:
            acts: Activation tensor (N, D)
            labels: True labels (N,)
        
        Returns:
            Dict with accuracy and auc metrics
        """
        if not self.is_fitted:
            raise ValueError("Probe not fitted. Call fit() first.")
        
        X = acts.cpu().numpy()
        y = labels.cpu().numpy()
        
        y_pred = self.lr.predict(X)
        y_proba = self.lr.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "auc": roc_auc_score(y, y_proba)
        }

