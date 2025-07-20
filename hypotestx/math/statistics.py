"""
Basic statistical functions implemented from scratch
"""
from typing import List, Optional, Tuple
from .basic import sqrt, power, abs_value

def mean(data: List[float]) -> float:
    """Calculate arithmetic mean"""
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(data) / len(data)

def median(data: List[float]) -> float:
    """Calculate median"""
    if not data:
        raise ValueError("Cannot calculate median of empty list")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

def mode(data: List[float]) -> List[float]:
    """Calculate mode(s)"""
    if not data:
        raise ValueError("Cannot calculate mode of empty list")
    
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    
    max_freq = max(frequency.values())
    modes = [value for value, freq in frequency.items() if freq == max_freq]
    
    return modes

def variance(data: List[float], ddof: int = 1) -> float:
    """Calculate variance with degrees of freedom correction"""
    if len(data) <= ddof:
        raise ValueError(f"Need at least {ddof + 1} data points")
    
    data_mean = mean(data)
    squared_diffs = [(x - data_mean) ** 2 for x in data]
    return sum(squared_diffs) / (len(data) - ddof)

def std(data: List[float], ddof: int = 1) -> float:
    """Calculate standard deviation"""
    return sqrt(variance(data, ddof))

def covariance(x: List[float], y: List[float], ddof: int = 1) -> float:
    """Calculate covariance between two variables"""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) <= ddof:
        raise ValueError(f"Need at least {ddof + 1} data points")
    
    x_mean = mean(x)
    y_mean = mean(y)
    
    cov_sum = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    return cov_sum / (len(x) - ddof)

def correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    
    cov = covariance(x, y)
    std_x = std(x)
    std_y = std(y)
    
    if std_x == 0 or std_y == 0:
        raise ValueError("Cannot calculate correlation with zero variance")
    
    return cov / (std_x * std_y)

def skewness(data: List[float]) -> float:
    """Calculate skewness (third moment)"""
    if len(data) < 3:
        raise ValueError("Need at least 3 data points for skewness")
    
    data_mean = mean(data)
    data_std = std(data)
    n = len(data)
    
    if data_std == 0:
        return 0.0
    
    skew_sum = sum(((x - data_mean) / data_std) ** 3 for x in data)
    return (n / ((n - 1) * (n - 2))) * skew_sum

def kurtosis(data: List[float]) -> float:
    """Calculate kurtosis (fourth moment)"""
    if len(data) < 4:
        raise ValueError("Need at least 4 data points for kurtosis")
    
    data_mean = mean(data)
    data_std = std(data)
    n = len(data)
    
    if data_std == 0:
        return 0.0
    
    kurt_sum = sum(((x - data_mean) / data_std) ** 4 for x in data)
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * kurt_sum
    kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    return kurt

def percentile(data: List[float], p: float) -> float:
    """Calculate percentile using linear interpolation"""
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if n == 1:
        return sorted_data[0]
    
    # Calculate the position
    pos = (p / 100) * (n - 1)
    
    if pos == int(pos):
        return sorted_data[int(pos)]
    
    # Linear interpolation
    lower = int(pos)
    upper = lower + 1
    weight = pos - lower
    
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

def quartiles(data: List[float]) -> Tuple[float, float, float]:
    """Calculate Q1, Q2 (median), Q3"""
    return (
        percentile(data, 25),
        percentile(data, 50),
        percentile(data, 75)
    )

def iqr(data: List[float]) -> float:
    """Calculate interquartile range"""
    q1, _, q3 = quartiles(data)
    return q3 - q1

def range_stat(data: List[float]) -> float:
    """Calculate range (max - min)"""
    if not data:
        raise ValueError("Cannot calculate range of empty list")
    return max(data) - min(data)

def mad(data: List[float]) -> float:
    """Calculate median absolute deviation"""
    if not data:
        raise ValueError("Cannot calculate MAD of empty list")
    
    data_median = median(data)
    absolute_deviations = [abs_value(x - data_median) for x in data]
    return median(absolute_deviations)

def trimmed_mean(data: List[float], trim_percent: float = 0.1) -> float:
    """Calculate trimmed mean"""
    if not 0 <= trim_percent < 0.5:
        raise ValueError("Trim percent must be between 0 and 0.5")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    trim_count = int(n * trim_percent)
    
    if trim_count * 2 >= n:
        raise ValueError("Too much trimming for the data size")
    
    trimmed_data = sorted_data[trim_count:n-trim_count]
    return mean(trimmed_data)