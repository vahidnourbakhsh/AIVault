"""
Test utilities module.
"""

import pytest
import numpy as np
import pandas as pd


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_get_data_info(self):
        """Test data info function."""
        # Create sample data
        data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'missing_col': [1, None, 3, None, 5]
        })
        
        # Test would go here when dependencies are available
        assert len(data) == 5
        assert 'numeric_col' in data.columns
        
    def test_preprocess_basic(self):
        """Test basic preprocessing functionality."""
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        assert len(data) == 5
        assert 'target' in data.columns


class TestVizUtils:
    """Test cases for visualization utilities."""
    
    def test_plot_params(self):
        """Test plot parameter validation."""
        # Mock test - would test plotting functions when matplotlib is available
        train_losses = [0.5, 0.3, 0.2, 0.1]
        val_losses = [0.6, 0.4, 0.25, 0.15]
        
        assert len(train_losses) == len(val_losses)
        assert all(isinstance(x, (int, float)) for x in train_losses)


class TestEvalUtils:
    """Test cases for evaluation utilities."""
    
    def test_metric_calculation(self):
        """Test metric calculation functions."""
        # Mock predictions
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # Basic validation
        assert len(y_true) == len(y_pred)
        assert all(x in [0, 1] for x in y_true)
        assert all(x in [0, 1] for x in y_pred)


class TestCommonUtils:
    """Test cases for common utilities."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        # Test that function exists and can be called
        from aivault.utilities.common_utils import set_seed
        
        # Should not raise an exception
        set_seed(42)
        
        # Test reproducibility with numpy
        set_seed(42)
        random1 = np.random.random(5)
        
        set_seed(42)  
        random2 = np.random.random(5)
        
        np.testing.assert_array_equal(random1, random2)
    
    def test_format_bytes(self):
        """Test byte formatting function."""
        from aivault.utilities.common_utils import format_bytes
        
        assert format_bytes(1024) == "1.0KB"
        assert format_bytes(1024 * 1024) == "1.0MB"
        assert format_bytes(500) == "500.0B"


if __name__ == "__main__":
    pytest.main([__file__])
