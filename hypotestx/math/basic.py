"""
Basic mathematical operations implemented from scratch
"""
import sys
from typing import List, Union, Optional

def abs_value(x: float) -> float:
    """Absolute value"""
    return x if x >= 0 else -x

def sqrt(x: float, precision: float = 1e-10) -> float:
    """Square root using Newton's method"""
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    if x == 0:
        return 0.0
    
    # Newton's method
    guess = x / 2.0
    while True:
        better_guess = (guess + x / guess) / 2.0
        if abs_value(better_guess - guess) < precision:
            return better_guess
        guess = better_guess

def exp(x: float, terms: int = 50) -> float:
    """Exponential function using Taylor series"""
    if x > 700:  # Prevent overflow
        return float('inf')
    if x < -700:
        return 0.0
    
    result = 1.0
    term = 1.0
    
    for i in range(1, terms):
        term *= x / i
        result += term
        if abs_value(term) < 1e-15:
            break
    
    return result

def ln(x: float, precision: float = 1e-10) -> float:
    """Natural logarithm using Newton's method"""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    if x == 1:
        return 0.0
    
    # Use the identity ln(x) = 2 * ln(sqrt(x)) to improve convergence
    if x > 2:
        return 2 * ln(sqrt(x))
    
    # Newton's method: x_{n+1} = x_n + 2 * (x - exp(x_n)) / (x + exp(x_n))
    guess = 0.0
    for _ in range(100):
        exp_guess = exp(guess)
        new_guess = guess + 2 * (x - exp_guess) / (x + exp_guess)
        if abs_value(new_guess - guess) < precision:
            return new_guess
        guess = new_guess
    
    return guess

def log(x: float, base: float = 10) -> float:
    """Logarithm with arbitrary base"""
    return ln(x) / ln(base)

def power(base: float, exponent: float) -> float:
    """Power function"""
    if base == 0:
        return 0.0 if exponent > 0 else float('inf')
    if exponent == 0:
        return 1.0
    if exponent == 1:
        return base
    
    # For integer exponents, use repeated multiplication
    if exponent == int(exponent):
        if exponent < 0:
            return 1.0 / power(base, -exponent)
        
        result = 1.0
        for _ in range(int(exponent)):
            result *= base
        return result
    
    # For fractional exponents, use exp(exponent * ln(base))
    if base < 0:
        raise ValueError("Cannot compute fractional power of negative number")
    
    return exp(exponent * ln(base))

def factorial(n: int) -> int:
    """Factorial function"""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combination(n: int, k: int) -> int:
    """Binomial coefficient C(n,k)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use the more efficient formula
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def sign(x: float) -> int:
    """Sign function"""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Constants
PI = 3.141592653589793
E = 2.718281828459045