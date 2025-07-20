from .basic import exp, ln, sqrt, abs_value, PI, factorial, power

def gamma(z: float) -> float:
    """
    Gamma function using multiple approaches for accuracy
    Γ(z) = (z-1)! for positive integers
    """
    
    # Handle exact integer cases first
    if z > 0 and abs(z - round(z)) < 1e-15:  # z is essentially an integer
        n = int(round(z))
        if n <= 20:  # Use factorial for small integers
            return float(factorial(n - 1))
    
    # Handle special values
    if abs(z - 0.5) < 1e-15:  # z = 0.5
        return sqrt(PI)
    
    if abs(z - 1.5) < 1e-15:  # z = 1.5
        return sqrt(PI) / 2
    
    if abs(z - 2.5) < 1e-15:  # z = 2.5
        return 3 * sqrt(PI) / 4
    
    # For negative values, use reflection formula
    if z < 0:
        if abs(z - round(z)) < 1e-15:  # Negative integer
            raise ValueError("Gamma function undefined for non-positive integers")
        return PI / (_sin_pi_times_x(z) * gamma(1 - z))
    
    # For z < 1, use recurrence relation: Γ(z+1) = z*Γ(z)
    if z < 1:
        return gamma(z + 1) / z
    
    # For large z, use Stirling's approximation
    if z > 12:
        return _stirling_approximation(z)
    
    # For 1 <= z <= 12, use Lanczos approximation
    return _lanczos_gamma(z)

def _sin_pi_times_x(x: float) -> float:
    """Compute sin(π*x) accurately for the reflection formula"""
    # Handle exact integer case
    if abs(x - round(x)) < 1e-15:
        return 0.0
    
    # Reduce to [0, 1) range
    x_reduced = x - int(x)
    if x_reduced < 0:
        x_reduced += 1
    
    # Use Taylor series for sin(πx)
    pi_x = PI * x_reduced
    result = pi_x
    term = pi_x
    
    for n in range(1, 20):
        term *= -(pi_x * pi_x) / ((2*n) * (2*n + 1))
        result += term
        if abs(term) < 1e-15:
            break
    
    return result

def _stirling_approximation(z: float) -> float:
    """Stirling's approximation for large z"""
    # Γ(z) ≈ √(2π/z) * (z/e)^z * (1 + 1/(12z) + 1/(288z²) + ...)
    
    log_gamma = (z - 0.5) * ln(z) - z + 0.5 * ln(2 * PI)
    
    # Add correction terms
    inv_z = 1.0 / z
    correction = inv_z / 12 - inv_z**3 / 360 + inv_z**5 / 1260
    log_gamma += correction
    
    return exp(log_gamma)

def _lanczos_gamma(z: float) -> float:
    """Lanczos approximation for gamma function"""
    # Use a simpler, more stable set of coefficients
    g = 7
    
    # Coefficients for Lanczos approximation
    coeff = [
        0.99999999999980993227684700473478,
        676.520368121885098567009190444019,
        -1259.13921672240287047156078755283,
        771.3234287776530788486528258894,
        -176.61502916214059906584551354,
        12.507343278686904814458936853,
        -0.13857109526572011689554707,
        9.9843695780195716571051946e-6,
        1.5056327351493116387e-7
    ]
    
    if z < 0.5:
        # Use reflection formula
        return PI / (_sin_pi_times_x(z) * _lanczos_gamma(1 - z))
    
    z -= 1
    x = coeff[0]
    for i in range(1, len(coeff)):
        x += coeff[i] / (z + i)
    
    t = z + g + 0.5
    sqrt_2pi = sqrt(2 * PI)
    
    result = sqrt_2pi * power(t, z + 0.5) * exp(-t) * x
    return result

def beta(a: float, b: float) -> float:
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
    # For integer values, use factorial formula when possible
    if (a > 0 and abs(a - round(a)) < 1e-15 and 
        b > 0 and abs(b - round(b)) < 1e-15):
        n = int(round(a))
        m = int(round(b))
        if n <= 20 and m <= 20 and (n + m) <= 20:
            return float(factorial(n-1) * factorial(m-1)) / float(factorial(n+m-1))
    
    return gamma(a) * gamma(b) / gamma(a + b)

def erf(x: float) -> float:
    """Error function using series expansion"""
    if x == 0:
        return 0.0
    
    # For large |x|, use asymptotic approximation
    if abs_value(x) > 4:
        return 1.0 if x > 0 else -1.0
    
    # For negative x, use erf(-x) = -erf(x)
    if x < 0:
        return -erf(-x)
    
    # Series expansion: erf(x) = (2/√π) * Σ((-1)^n * x^(2n+1))/(n!(2n+1))
    coefficient = 2.0 / sqrt(PI)
    term = x
    result = term
    
    for n in range(1, 100):
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
    if x == 0:
        return 0.0
    
    # Use series representation for small x relative to a
    if x < a + 1:
        return _gamma_series(a, x)
    else:
        # Use continued fraction for large x
        return gamma(a) - _gamma_continued_fraction(a, x)

def _gamma_series(a: float, x: float) -> float:
    """Series representation of incomplete gamma"""
    if x == 0:
        return 0.0
    
    # Series: γ(a,x) = x^a * e^(-x) * Σ(x^n / Γ(a+n+1)) from n=0 to ∞
    # = x^a * e^(-x) / Γ(a) * Σ(x^n / (a)_n) where (a)_n = a(a+1)...(a+n-1)
    
    term = 1.0 / a
    result = term
    
    for n in range(1, 200):
        term *= x / (a + n)
        result += term
        
        if abs_value(term) < 1e-15:
            break
    
    return result * power(x, a) * exp(-x)

def _gamma_continued_fraction(a: float, x: float) -> float:
    """Continued fraction representation of incomplete gamma complement"""
    # This computes Γ(a,x) = Γ(a) - γ(a,x)
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