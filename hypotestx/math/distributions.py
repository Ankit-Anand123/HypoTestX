from typing import List, Optional
from .basic import exp, ln, sqrt, power, PI, factorial, combination
from .special import gamma, beta, erf, gamma_incomplete, beta_incomplete

class Distribution:
    """Base class for probability distributions"""
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        raise NotImplementedError
    
    def cdf(self, x: float) -> float:
        """Cumulative distribution function"""
        raise NotImplementedError
    
    def ppf(self, p: float) -> float:
        """Percent point function (inverse CDF)"""
        raise NotImplementedError

class Normal(Distribution):
    """Normal distribution"""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x: float) -> float:
        """Normal probability density function"""
        coefficient = 1.0 / (self.sigma * sqrt(2 * PI))
        exponent = -0.5 * power((x - self.mu) / self.sigma, 2)
        return coefficient * exp(exponent)
    
    def cdf(self, x: float) -> float:
        """Normal cumulative distribution function"""
        # Use error function
        z = (x - self.mu) / (self.sigma * sqrt(2))
        return 0.5 * (1 + erf(z))
    
    def ppf(self, p: float) -> float:
        """Normal percent point function (inverse CDF)"""
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        
        # Approximate inverse using Beasley-Springer-Moro algorithm
        return self._inverse_normal_cdf(p)
    
    def _inverse_normal_cdf(self, p: float) -> float:
        """Approximate inverse normal CDF"""
        # Beasley-Springer-Moro approximation
        a0 = 2.50662823884
        a1 = -18.61500062529
        a2 = 41.39119773534
        a3 = -25.44106049637
        
        b1 = -8.47351093090
        b2 = 23.08336743743
        b3 = -21.06224101826
        b4 = 3.13082909833
        
        c0 = 0.3374754822726147
        c1 = 0.9761690190917186
        c2 = 0.1607979714918209
        c3 = 0.0276438810333863
        c4 = 0.0038405729373609
        c5 = 0.0003951896511919
        c6 = 0.0000321767881768
        c7 = 0.0000002888167364
        c8 = 0.0000003960315187
        
        if p < 0.5:
            # Use the fact that inverse_normal_cdf(p) = -inverse_normal_cdf(1-p)
            return -self._inverse_normal_cdf(1 - p)
        
        # For p >= 0.5
        if p == 0.5:
            return 0.0
        
        r = sqrt(-2 * ln(1 - p))
        
        if r <= 5.0:
            r -= 1.6
            result = (((a3 * r + a2) * r + a1) * r + a0) / \
                    ((((b4 * r + b3) * r + b2) * r + b1) * r + 1)
        else:
            r -= 5.0
            result = (((((((c8 * r + c7) * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0
        
        return self.mu + self.sigma * result

class StudentT(Distribution):
    """Student's t-distribution"""
    
    def __init__(self, df: float):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
    
    def pdf(self, x: float) -> float:
        """t-distribution probability density function"""
        coefficient = gamma((self.df + 1) / 2) / (sqrt(self.df * PI) * gamma(self.df / 2))
        factor = power(1 + x * x / self.df, -(self.df + 1) / 2)
        return coefficient * factor
    
    def cdf(self, x: float) -> float:
        """t-distribution cumulative distribution function"""
        # For large df, approximate with normal
        if self.df > 100:
            normal = Normal(0, 1)
            return normal.cdf(x)
        
        # Use incomplete beta function
        if x == 0:
            return 0.5
        
        t = self.df / (self.df + x * x)
        result = 0.5 * beta_incomplete(self.df / 2, 0.5, t)
        
        if x > 0:
            return 1 - result
        else:
            return result
    
    def ppf(self, p: float) -> float:
        """t-distribution percent point function (inverse CDF)"""
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        
        # For large df, use normal approximation
        if self.df > 100:
            normal = Normal(0, 1)
            return normal.ppf(p)
        
        # Use iterative method to find inverse
        return self._inverse_t_cdf(p)
    
    def _inverse_t_cdf(self, p: float) -> float:
        """Approximate inverse t-distribution CDF using bisection"""
        if p == 0.5:
            return 0.0
        
        # Initial bounds
        if p < 0.5:
            lower, upper = -10.0, 0.0
        else:
            lower, upper = 0.0, 10.0
        
        # Bisection method
        for _ in range(100):
            mid = (lower + upper) / 2
            cdf_mid = self.cdf(mid)
            
            if abs(cdf_mid - p) < 1e-10:
                return mid
            
            if cdf_mid < p:
                lower = mid
            else:
                upper = mid
        
        return (lower + upper) / 2

class ChiSquare(Distribution):
    """Chi-square distribution"""
    
    def __init__(self, df: float):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
    
    def pdf(self, x: float) -> float:
        """Chi-square probability density function"""
        if x < 0:
            return 0.0
        
        if x == 0:
            if self.df < 2:
                return float('inf')
            elif self.df == 2:
                return 0.5  # For df=2, PDF at x=0 is 1/2
            else:
                return 0.0  # For df>2, PDF at x=0 is 0
        
        # For x > 0, use the standard formula
        try:
            coefficient = 1 / (power(2, self.df / 2) * gamma(self.df / 2))
            return coefficient * power(x, self.df / 2 - 1) * exp(-x / 2)
        except (OverflowError, ZeroDivisionError):
            return 0.0
    
    def cdf(self, x: float) -> float:
        """Chi-square cumulative distribution function"""
        if x <= 0:
            return 0.0
        
        # Use incomplete gamma function
        try:
            return gamma_incomplete(self.df / 2, x / 2) / gamma(self.df / 2)
        except (OverflowError, ZeroDivisionError):
            return 1.0 if x > self.df + 10 * sqrt(2 * self.df) else 0.0
    
    def ppf(self, p: float) -> float:
        """Chi-square percent point function (inverse CDF)"""
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        
        # Use iterative method to find inverse
        return self._inverse_chi2_cdf(p)
    
    def _inverse_chi2_cdf(self, p: float) -> float:
        """Approximate inverse chi-square CDF using bisection"""
        # Initial bounds
        lower, upper = 0.0, self.df + 6 * sqrt(2 * self.df)
        
        # Bisection method
        for _ in range(100):
            mid = (lower + upper) / 2
            cdf_mid = self.cdf(mid)
            
            if abs(cdf_mid - p) < 1e-10:
                return mid
            
            if cdf_mid < p:
                lower = mid
            else:
                upper = mid
        
        return (lower + upper) / 2

class F(Distribution):
    """F-distribution"""
    
    def __init__(self, df1: float, df2: float):
        if df1 <= 0 or df2 <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df1 = df1
        self.df2 = df2
    
    def pdf(self, x: float) -> float:
        """F-distribution probability density function"""
        if x <= 0:
            return 0.0
        
        try:
            coefficient = (gamma((self.df1 + self.df2) / 2) / 
                          (gamma(self.df1 / 2) * gamma(self.df2 / 2)))
            coefficient *= power(self.df1 / self.df2, self.df1 / 2)
            coefficient *= power(x, self.df1 / 2 - 1)
            denominator = power(1 + (self.df1 / self.df2) * x, (self.df1 + self.df2) / 2)
            
            return coefficient / denominator
        except (OverflowError, ZeroDivisionError):
            return 0.0
    
    def cdf(self, x: float) -> float:
        """F-distribution cumulative distribution function"""
        if x <= 0:
            return 0.0
        
        try:
            # Use incomplete beta function
            t = self.df1 * x / (self.df1 * x + self.df2)
            return beta_incomplete(self.df1 / 2, self.df2 / 2, t)
        except (OverflowError, ZeroDivisionError):
            return 1.0 if x > 10 else 0.0
    
    def ppf(self, p: float) -> float:
        """F-distribution percent point function (inverse CDF)"""
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        
        # Use iterative method to find inverse
        return self._inverse_f_cdf(p)
    
    def _inverse_f_cdf(self, p: float) -> float:
        """Approximate inverse F-distribution CDF using bisection"""
        # Initial bounds
        lower, upper = 0.0, 10.0
        
        # Expand upper bound if necessary
        while self.cdf(upper) < p:
            upper *= 2
        
        # Bisection method
        for _ in range(100):
            mid = (lower + upper) / 2
            cdf_mid = self.cdf(mid)
            
            if abs(cdf_mid - p) < 1e-10:
                return mid
            
            if cdf_mid < p:
                lower = mid
            else:
                upper = mid
        
        return (lower + upper) / 2