"""
HypoTestX: Natural Language Hypothesis Testing Library
"""

from .core.parser import parse_hypothesis, create_parser
from .core.result import HypoResult
from .tests.parametric import one_sample_ttest, two_sample_ttest, paired_ttest
from .math.statistics import mean, std, variance, correlation
from .math.distributions import Normal, StudentT, ChiSquare, F

__version__ = "0.1.0"
__author__ = "Your Name"

def test(hypothesis: str, data=None, **kwargs):
    """
    Main natural language interface for hypothesis testing
    
    Args:
        hypothesis: Natural language hypothesis statement
        data: Optional pandas DataFrame or similar data structure
        **kwargs: Additional parameters for specific tests
    
    Returns:
        HypoResult object with test results
    
    Examples:
        >>> import hypotestx as htx
        >>> result = htx.test("Do males spend more than females?", data=df)
        >>> print(result.summary())
        
        >>> result = htx.test("Is the mean different from 100?", data=df['values'])
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    # Parse the hypothesis
    parsed = parse_hypothesis(hypothesis, data)
    
    # Extract parameters
    alpha = kwargs.get('alpha', parsed.confidence_level)
    alternative = kwargs.get('alternative', parsed.tail)
    
    # Route to appropriate test
    if parsed.test_type == "one_sample_ttest":
        if data is None:
            raise ValueError("Data is required for statistical testing")
        
        # Extract data values
        if hasattr(data, 'values'):  # pandas Series or similar
            values = list(data.values)
        elif isinstance(data, list):
            values = data
        else:
            raise ValueError("Unsupported data format")
        
        mu = kwargs.get('mu', 0.0)
        return one_sample_ttest(values, mu=mu, alpha=alpha, alternative=alternative)
    
    elif parsed.test_type == "two_sample_ttest":
        if data is None or parsed.group_column is None or parsed.value_column is None:
            raise ValueError("Data with group and value columns required")
        
        # Extract groups
        unique_groups = list(data[parsed.group_column].unique())
        if len(unique_groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(unique_groups)}")
        
        group1_data = data[data[parsed.group_column] == unique_groups[0]][parsed.value_column].values
        group2_data = data[data[parsed.group_column] == unique_groups[1]][parsed.value_column].values
        
        equal_var = kwargs.get('equal_var', True)
        return two_sample_ttest(
            list(group1_data), 
            list(group2_data), 
            alpha=alpha, 
            alternative=alternative, 
            equal_var=equal_var
        )
    
    elif parsed.test_type == "paired_ttest":
        if 'before' not in kwargs or 'after' not in kwargs:
            raise ValueError("Paired t-test requires 'before' and 'after' data")
        
        return paired_ttest(kwargs['before'], kwargs['after'], alpha=alpha, alternative=alternative)
    
    else:
        raise NotImplementedError(f"Test type '{parsed.test_type}' not yet implemented")

# Convenience functions
def ttest_1samp(data, mu=0.0, **kwargs):
    """One-sample t-test convenience function"""
    return one_sample_ttest(data, mu=mu, **kwargs)

def ttest_2samp(group1, group2, **kwargs):
    """Two-sample t-test convenience function"""
    return two_sample_ttest(group1, group2, **kwargs)

def ttest_paired(before, after, **kwargs):
    """Paired t-test convenience function"""
    return paired_ttest(before, after, **kwargs)

# Export main components
__all__ = [
    'test',
    'HypoResult', 
    'parse_hypothesis',
    'ttest_1samp',
    'ttest_2samp', 
    'ttest_paired',
    'one_sample_ttest',
    'two_sample_ttest',
    'paired_ttest',
    'mean',
    'std',
    'variance',
    'correlation',
    'Normal',
    'StudentT',
    'ChiSquare',
    'F'
]