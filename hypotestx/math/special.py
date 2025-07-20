"""
Special mathematical functions implemented from scratch
"""
from .basic import exp, ln, sqrt, abs_value, PI, factorial

def gamma(z: float, precision: float = 1e-10) -> float:
    """Gamma function using Lanczos approximation"""
    if z <= 0:
        raise ValueError("Gamma function undefined for non-positive integers")
    
    # Lanczos coefficients for g=7
    g = 7
    coeff = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    
    z -= 1
    x = coeff[0]
    for i in range(1, g + 2):
        x += coeff[i] / (z + i)
    
    t = z + g + 0.5
    sqrt_2pi = sqrt(2 * PI)
    
    return sqrt_2pi * power(t, z + 0.5) * exp(-t) * x

def beta(a: float, b: float) -> float:
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
    return gamma(a) * gamma(b) / gamma(a + b)

def erf(x: float) -> float:
    """Error function using series expansion"""
    if x == 0:
        return 0.0
    
    # For large |x|, use asymptotic approximation
    if abs_value(x) > 3:
        return 1.0 if x > 0 else -1.0
    
    # Series expansion: erf(x) = (2/√π) * Σ((-1)^n * x^(2n+1))/(n!(2n+1))
    coefficient = 2.0 / sqrt(PI)
    term = x
    result = term
    
    for n in range(1, 50):
        term *= -x * x / n
        new_term = term / (2 * n + 1)
        result += new_term
        
        if abs_value(new_term) < 1e-15:
            break
    
    return coefficient * result

def gamma_incomplete(a: float, x: float) -> float:
    """Lower incomplete gamma function"""
    if a <= 0:
        raise ValueError("Parameter 'a' must be positive")
    if x < 0:
        return 0.0
    
    # Use series representation for small x
    if x < a + 1:
        return _gamma_series(a, x)
    else:
        # Use continued fraction for large x
        return gamma(a) - _gamma_continued_fraction(a, x)

def _gamma_series(a: float, x: float) -> float:
    """Series representation of incomplete gamma"""
    if x == 0:
        return 0.0
    
    term = 1.0 / a
    result = term
    
    for n in range(1, 200):
        term *= x / (a + n)
        result += term
        
        if abs_value(term) < 1e-15:
            break
    
    return result * power(x, a) * exp(-x)

def _gamma_continued_fraction(a: float, x: float) -> float:
    """Continued fraction representation of incomplete gamma"""
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        
        if abs_value(d) < 1e-30:
            d = 1e-30
        
        c = b + an / c
        if abs_value(c) < 1e-30:
            c = 1e-30
        
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        
        if abs_value(del_h - 1.0) < 1e-15:
            break
    
    return h * power(x, a) * exp(-x)

def beta_incomplete(a: float, b: float, x: float) -> float:
    """Incomplete beta function"""
    if x < 0 or x > 1:
        raise ValueError("x must be between 0 and 1")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    
    # Use continued fraction
    bt = exp(a * ln(x) + b * ln(1 - x) - ln(beta(a, b)))
    
    if x < (a + 1) / (a + b + 2):
        return bt * _beta_continued_fraction(a, b, x) / a
    else:
        return 1.0 - bt * _beta_continued_fraction(b, a, 1 - x) / b

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    
    if abs_value(d) < 1e-30:
        d = 1e-30
    
    d = 1.0 / d
    h = d
    
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        
        if abs_value(d) < 1e-30:
            d = 1e-30
        
        c = 1.0 + aa / c
        if abs_value(c) < 1e-30:
            c = 1e-30
        
        d = 1.0 / d
        h *= d * c
        
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        
        if abs_value(d) < 1e-30:
            d = 1e-30
        
        c = 1.0 + aa / c
        if abs_value(c) < 1e-30:
            c = 1e-30
        
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        
        if abs_value(del_h - 1.0) < 1e-15:
            break
    
    return h