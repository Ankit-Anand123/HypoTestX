"""
Utility functions for formatting statistical output
"""

def format_p_value(p_value: float, threshold: float = 0.001) -> str:
    """Format p-value for display"""
    if p_value < threshold:
        return f"< {threshold}"
    elif p_value < 0.01:
        return f"{p_value:.4f}"
    elif p_value < 0.05:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.3f}"

def format_effect_size(effect_size: float, effect_type: str = "Cohen's d") -> str:
    """Format effect size for display"""
    if effect_type == "Cohen's d":
        if abs(effect_size) < 0.2:
            magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            magnitude = "small"
        elif abs(effect_size) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"{effect_size:.3f} ({magnitude})"
    
    return f"{effect_size:.3f}"

def format_confidence_interval(ci: tuple, confidence_level: float = 0.95) -> str:
    """Format confidence interval for display"""
    percentage = int(confidence_level * 100)
    return f"{percentage}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"