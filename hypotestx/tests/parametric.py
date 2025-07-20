from typing import List, Optional, Tuple, Union
from ..math.statistics import mean, std, variance
from ..math.distributions import Normal, StudentT
from ..math.basic import sqrt, abs_value
from ..core.result import HypoResult

def one_sample_ttest(
    data: List[float],
    mu: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> HypoResult:
    """
    One-sample t-test implemented from scratch
    
    Args:
        data: Sample data
        mu: Hypothesized population mean
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"
    
    Returns:
        HypoResult object with test results
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for t-test")
    
    n = len(data)
    sample_mean = mean(data)
    sample_std = std(data, ddof=1)
    
    if sample_std == 0:
        raise ValueError("Sample standard deviation is zero")
    
    # Calculate t-statistic
    standard_error = sample_std / sqrt(n)
    t_stat = (sample_mean - mu) / standard_error
    
    # Degrees of freedom
    df = n - 1
    
    # Calculate p-value
    t_dist = StudentT(df)
    
    if alternative == "two-sided":
        p_value = 2 * (1 - t_dist.cdf(abs_value(t_stat)))
    elif alternative == "greater":
        p_value = 1 - t_dist.cdf(t_stat)
    elif alternative == "less":
        p_value = t_dist.cdf(t_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Calculate effect size (Cohen's d)
    cohens_d = (sample_mean - mu) / sample_std
    
    # Calculate confidence interval
    t_critical = t_dist.ppf(1 - alpha/2) if alternative == "two-sided" else t_dist.ppf(1 - alpha)
    margin_of_error = t_critical * standard_error
    
    if alternative == "two-sided":
        ci = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    elif alternative == "greater":
        ci = (sample_mean - margin_of_error, float('inf'))
    else:  # less
        ci = (float('-inf'), sample_mean + margin_of_error)
    
    # Data summary
    data_summary = {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "sample_size": n,
        "hypothesized_mean": mu,
        "standard_error": standard_error
    }
    
    # Generate interpretation
    significance = "significant" if p_value < alpha else "not significant"
    direction = ""
    if alternative == "greater" and p_value < alpha:
        direction = f"Sample mean ({sample_mean:.3f}) is significantly greater than {mu}"
    elif alternative == "less" and p_value < alpha:
        direction = f"Sample mean ({sample_mean:.3f}) is significantly less than {mu}"
    elif alternative == "two-sided" and p_value < alpha:
        direction = f"Sample mean ({sample_mean:.3f}) is significantly different from {mu}"
    else:
        direction = f"No significant difference found between sample mean ({sample_mean:.3f}) and {mu}"
    
    interpretation = f"The one-sample t-test is {significance} (p = {p_value:.4f}). {direction}"
    
    return HypoResult(
        test_name="One-Sample t-test",
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        confidence_interval=ci,
        degrees_of_freedom=df,
        sample_sizes=n,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary
    )

def two_sample_ttest(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    equal_var: bool = True
) -> HypoResult:
    """
    Two-sample t-test (Student's t-test or Welch's t-test)
    
    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"
        equal_var: Whether to assume equal variances (Student's vs Welch's)
    
    Returns:
        HypoResult object with test results
    """
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Need at least 2 data points in each group")
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = mean(group1), mean(group2)
    var1, var2 = variance(group1, ddof=1), variance(group2, ddof=1)
    std1, std2 = sqrt(var1), sqrt(var2)
    
    # Calculate pooled or separate variance
    if equal_var:
        # Student's t-test (pooled variance)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        standard_error = sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
        test_name = "Student's t-test (equal variances)"
        pooled_std = sqrt(pooled_var)
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        # Welch's t-test (unequal variances)
        standard_error = sqrt(var1/n1 + var2/n2)
        # Welch-Satterthwaite equation for degrees of freedom
        numerator = (var1/n1 + var2/n2) ** 2
        denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = numerator / denominator
        test_name = "Welch's t-test (unequal variances)"
        cohens_d = (mean1 - mean2) / sqrt((var1 + var2) / 2)
    
    # Calculate t-statistic
    t_stat = (mean1 - mean2) / standard_error
    
    # Calculate p-value
    t_dist = StudentT(df)
    
    if alternative == "two-sided":
        p_value = 2 * (1 - t_dist.cdf(abs_value(t_stat)))
    elif alternative == "greater":
        p_value = 1 - t_dist.cdf(t_stat)
    elif alternative == "less":
        p_value = t_dist.cdf(t_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Calculate confidence interval for difference in means
    t_critical = t_dist.ppf(1 - alpha/2) if alternative == "two-sided" else t_dist.ppf(1 - alpha)
    margin_of_error = t_critical * standard_error
    mean_diff = mean1 - mean2
    
    if alternative == "two-sided":
        ci = (mean_diff - margin_of_error, mean_diff + margin_of_error)
    elif alternative == "greater":
        ci = (mean_diff - margin_of_error, float('inf'))
    else:  # less
        ci = (float('-inf'), mean_diff + margin_of_error)
    
    # Data summary
    data_summary = {
        "group1_mean": mean1,
        "group1_std": std1,
        "group1_size": n1,
        "group2_mean": mean2,
        "group2_std": std2,
        "group2_size": n2,
        "mean_difference": mean_diff,
        "standard_error": standard_error,
        "pooled_variance": pooled_var if equal_var else None
    }
    
    # Generate interpretation
    significance = "significant" if p_value < alpha else "not significant"
    direction = ""
    if alternative == "greater" and p_value < alpha:
        direction = f"Group 1 mean ({mean1:.3f}) is significantly greater than Group 2 mean ({mean2:.3f})"
    elif alternative == "less" and p_value < alpha:
        direction = f"Group 1 mean ({mean1:.3f}) is significantly less than Group 2 mean ({mean2:.3f})"
    elif alternative == "two-sided" and p_value < alpha:
        direction = f"Significant difference between Group 1 mean ({mean1:.3f}) and Group 2 mean ({mean2:.3f})"
    else:
        direction = f"No significant difference between Group 1 mean ({mean1:.3f}) and Group 2 mean ({mean2:.3f})"
    
    interpretation = f"The {test_name.lower()} is {significance} (p = {p_value:.4f}). {direction}"
    
    return HypoResult(
        test_name=test_name,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        confidence_interval=ci,
        degrees_of_freedom=df,
        sample_sizes=(n1, n2),
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary
    )

def paired_ttest(
    before: List[float],
    after: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> HypoResult:
    """
    Paired t-test for dependent samples
    
    Args:
        before: Before measurements
        after: After measurements  
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"
    
    Returns:
        HypoResult object with test results
    """
    if len(before) != len(after):
        raise ValueError("Before and after groups must have same length")
    
    if len(before) < 2:
        raise ValueError("Need at least 2 paired observations")
    
    # Calculate differences
    differences = [after[i] - before[i] for i in range(len(before))]
    
    # Use one-sample t-test on differences
    result = one_sample_ttest(differences, mu=0.0, alpha=alpha, alternative=alternative)
    
    # Update test name and interpretation
    result.test_name = "Paired t-test"
    
    n = len(differences)
    mean_diff = mean(differences)
    std_diff = std(differences, ddof=1)
    
    # Update data summary
    result.data_summary.update({
        "before_mean": mean(before),
        "after_mean": mean(after),
        "before_std": std(before, ddof=1),
        "after_std": std(after, ddof=1),
        "mean_difference": mean_diff,
        "difference_std": std_diff,
        "n_pairs": n
    })
    
    # Update interpretation
    significance = "significant" if result.p_value < alpha else "not significant"
    if alternative == "greater" and result.p_value < alpha:
        direction = f"After values are significantly greater than before values (mean difference = {mean_diff:.3f})"
    elif alternative == "less" and result.p_value < alpha:
        direction = f"After values are significantly less than before values (mean difference = {mean_diff:.3f})"
    elif alternative == "two-sided" and result.p_value < alpha:
        direction = f"Significant difference between before and after values (mean difference = {mean_diff:.3f})"
    else:
        direction = f"No significant difference between before and after values (mean difference = {mean_diff:.3f})"
    
    result.interpretation = f"The paired t-test is {significance} (p = {result.p_value:.4f}). {direction}"
    
    return result