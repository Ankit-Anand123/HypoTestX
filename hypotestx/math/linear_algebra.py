from typing import List, Tuple, Optional, Union
from .basic import sqrt, abs_value, sign

class Matrix:
    """Matrix class with basic operations"""
    
    def __init__(self, data: List[List[float]]):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        
        # Validate rectangular matrix
        rows = len(data)
        cols = len(data[0])
        for row in data:
            if len(row) != cols:
                raise ValueError("All rows must have same number of columns")
        
        self.data = [row[:] for row in data]  # Deep copy
        self.rows = rows
        self.cols = cols
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i][j]
        return self.data[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            self.data[i][j] = value
        else:
            self.data[key] = value
    
    def __str__(self):
        return '\n'.join([' '.join(f'{x:8.4f}' for x in row) for row in self.data])
    
    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for addition")
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for subtraction")
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.data[i][j] * other)
                result.append(row)
            return Matrix(result)
        else:
            # Matrix multiplication
            return matrix_multiply(self, other)
    
    def transpose(self):
        """Return transpose of matrix"""
        result = []
        for j in range(self.cols):
            row = []
            for i in range(self.rows):
                row.append(self.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def is_square(self) -> bool:
        """Check if matrix is square"""
        return self.rows == self.cols
    
    def determinant(self) -> float:
        """Calculate determinant for square matrices"""
        if not self.is_square():
            raise ValueError("Determinant only defined for square matrices")
        
        return _determinant_recursive(self.data)
    
    def trace(self) -> float:
        """Calculate trace (sum of diagonal elements)"""
        if not self.is_square():
            raise ValueError("Trace only defined for square matrices")
        
        return sum(self.data[i][i] for i in range(self.rows))

def vector_dot(a: List[float], b: List[float]) -> float:
    """Calculate dot product of two vectors"""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    
    return sum(a[i] * b[i] for i in range(len(a)))

def vector_norm(vector: List[float], p: int = 2) -> float:
    """Calculate p-norm of vector (default: Euclidean norm)"""
    if p == 1:
        return sum(abs_value(x) for x in vector)
    elif p == 2:
        return sqrt(sum(x * x for x in vector))
    elif p == float('inf'):
        return max(abs_value(x) for x in vector)
    else:
        return sum(abs_value(x) ** p for x in vector) ** (1.0 / p)

def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """Multiply two matrices"""
    if A.cols != B.rows:
        raise ValueError(f"Cannot multiply {A.rows}x{A.cols} and {B.rows}x{B.cols} matrices")
    
    result = []
    for i in range(A.rows):
        row = []
        for j in range(B.cols):
            # Dot product of row i of A and column j of B
            value = 0.0
            for k in range(A.cols):
                value += A.data[i][k] * B.data[k][j]
            row.append(value)
        result.append(row)
    
    return Matrix(result)

def matrix_transpose(matrix: Matrix) -> Matrix:
    """Return transpose of matrix"""
    return matrix.transpose()

def matrix_inverse(matrix: Matrix) -> Matrix:
    """Calculate matrix inverse using Gauss-Jordan elimination"""
    if not matrix.is_square():
        raise ValueError("Only square matrices can be inverted")
    
    n = matrix.rows
    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = matrix.data[i][:] + [0.0] * n
        row[n + i] = 1.0  # Identity matrix
        augmented.append(row)
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs_value(augmented[k][i]) > abs_value(augmented[max_row][i]):
                max_row = k
        
        if abs_value(augmented[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular (not invertible)")
        
        # Swap rows
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Scale pivot row
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract inverse matrix
    inverse_data = []
    for i in range(n):
        inverse_data.append(augmented[i][n:])
    
    return Matrix(inverse_data)

def _determinant_recursive(matrix: List[List[float]]) -> float:
    """Calculate determinant recursively"""
    n = len(matrix)
    
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0.0
    for j in range(n):
        # Create minor matrix
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            minor.append(row)
        
        # Calculate cofactor
        cofactor = ((-1) ** j) * matrix[0][j] * _determinant_recursive(minor)
        det += cofactor
    
    return det

def eigenvalues(matrix: Matrix, max_iterations: int = 1000) -> List[float]:
    """Calculate eigenvalues using QR algorithm"""
    if not matrix.is_square():
        raise ValueError("Eigenvalues only defined for square matrices")
    
    # QR algorithm for eigenvalues
    A = Matrix([row[:] for row in matrix.data])  # Copy
    
    for _ in range(max_iterations):
        Q, R = qr_decomposition(A)
        A = matrix_multiply(R, Q)
    
    # Extract eigenvalues from diagonal
    eigenvals = []
    for i in range(A.rows):
        eigenvals.append(A.data[i][i])
    
    return eigenvals

def eigenvectors(matrix: Matrix, eigenvals: List[float]) -> List[List[float]]:
    """Calculate eigenvectors for given eigenvalues"""
    if not matrix.is_square():
        raise ValueError("Eigenvectors only defined for square matrices")
    
    vectors = []
    n = matrix.rows
    
    for eigenval in eigenvals:
        # Solve (A - λI)v = 0
        A_minus_lambdaI = matrix - Matrix([[eigenval if i == j else 0.0 
                                          for j in range(n)] for i in range(n)])
        
        # Find null space vector using power iteration
        v = [1.0] + [0.0] * (n - 1)  # Initial guess
        
        for _ in range(100):  # Power iteration
            # Multiply by (A - λI)
            new_v = [0.0] * n
            for i in range(n):
                for j in range(n):
                    new_v[i] += A_minus_lambdaI.data[i][j] * v[j]
            
            # Normalize
            norm = vector_norm(new_v)
            if norm > 1e-12:
                v = [x / norm for x in new_v]
            else:
                break
        
        vectors.append(v)
    
    return vectors

def qr_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Gram-Schmidt process"""
    m, n = matrix.rows, matrix.cols
    
    # Initialize Q and R
    Q_data = [[0.0] * n for _ in range(m)]
    R_data = [[0.0] * n for _ in range(n)]
    
    # Gram-Schmidt process
    for j in range(n):
        # Get column j
        col_j = [matrix.data[i][j] for i in range(m)]
        
        # Orthogonalize against previous columns
        for k in range(j):
            # R[k,j] = Q_k^T * A_j
            R_data[k][j] = sum(Q_data[i][k] * matrix.data[i][j] for i in range(m))
            
            # col_j = col_j - R[k,j] * Q_k
            for i in range(m):
                col_j[i] -= R_data[k][j] * Q_data[i][k]
        
        # Normalize
        R_data[j][j] = vector_norm(col_j)
        
        if R_data[j][j] > 1e-12:
            for i in range(m):
                Q_data[i][j] = col_j[i] / R_data[j][j]
    
    return Matrix(Q_data), Matrix(R_data)

def svd_decomposition(matrix: Matrix, max_iterations: int = 1000) -> Tuple[Matrix, List[float], Matrix]:
    """
    Singular Value Decomposition using iterative algorithm
    Returns U, S, V such that A = U * S * V^T
    """
    m, n = matrix.rows, matrix.cols
    
    # For simplicity, implement a basic SVD using eigendecomposition
    # A^T * A has eigenvalues σ²
    AT = matrix.transpose()
    ATA = matrix_multiply(AT, matrix)
    
    # Get eigenvalues and eigenvectors of A^T * A
    eigenvals_ATA = eigenvalues(ATA, max_iterations)
    eigenvecs_ATA = eigenvectors(ATA, eigenvals_ATA)
    
    # Singular values are square roots of eigenvalues
    singular_values = [sqrt(max(0, val)) for val in eigenvals_ATA]
    
    # V is the matrix of eigenvectors of A^T * A
    V_data = []
    for i in range(n):
        V_data.append(eigenvecs_ATA[i])
    V = Matrix(V_data).transpose()
    
    # U can be computed as A * V * S^(-1)
    U_data = [[0.0] * min(m, n) for _ in range(m)]
    
    for j in range(min(m, n)):
        if singular_values[j] > 1e-12:
            # u_j = (1/σ_j) * A * v_j
            v_j = [V.data[i][j] for i in range(n)]
            Av_j = [0.0] * m
            
            for i in range(m):
                for k in range(n):
                    Av_j[i] += matrix.data[i][k] * v_j[k]
            
            for i in range(m):
                U_data[i][j] = Av_j[i] / singular_values[j]
    
    U = Matrix(U_data)
    
    return U, singular_values, V