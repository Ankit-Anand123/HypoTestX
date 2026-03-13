"""
Basic mathematical operations implemented from scratch
"""


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


def exp(x: float, terms: int = 100) -> float:
    """Exponential function using Taylor series.

    For negative *x* we use the identity exp(-x) = 1/exp(x) to avoid
    catastrophic cancellation in the alternating Taylor series.
    """
    if x > 700:  # Prevent overflow
        return float("inf")
    if x < -700:
        return 0.0

    # Negative argument: compute exp(-x) via reciprocal to stay numerically stable
    if x < 0.0:
        return 1.0 / exp(-x)

    # Range-reduce: exp(x) = exp(k + r) = e^k * exp(r)  with 0 <= r < 1
    # We use repeated squaring for the integer part.
    k = int(x)
    r = x - k  # fractional remainder in [0, 1)

    # Taylor series for exp(r), 0 <= r < 1 — converges very quickly
    result = 1.0
    term = 1.0
    for i in range(1, terms):
        term *= r / i
        result += term
        if abs_value(term) < 1e-17:
            break

    # Multiply by e^k using integer repeated multiplication of e
    # e ≈ 2.718281828459045235360287
    E = 2.718281828459045235360287
    ek = 1.0
    base = E
    n = k
    while n > 0:
        if n & 1:
            ek *= base
        base *= base
        n >>= 1

    return result * ek


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
        return 0.0 if exponent > 0 else float("inf")
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
