"""
Tests for hypotestx.math.linear_algebra.
All matrix functions expect Matrix objects, returning Matrix objects.
"""

import math

import pytest

from hypotestx.math.linear_algebra import (
    Matrix,
    matrix_inverse,
    matrix_multiply,
    matrix_transpose,
    qr_decomposition,
    vector_dot,
    vector_norm,
)


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def mat_approx(A: Matrix, B: Matrix, tol=1e-6):
    if A.rows != B.rows or A.cols != B.cols:
        return False
    for i in range(A.rows):
        for j in range(A.cols):
            if abs(A.data[i][j] - B.data[i][j]) > tol:
                return False
    return True


class TestVectorDot:
    def test_basic(self):
        assert approx(vector_dot([1, 2, 3], [4, 5, 6]), 32.0)

    def test_orthogonal(self):
        assert approx(vector_dot([1, 0], [0, 1]), 0.0)

    def test_self(self):
        v = [3.0, 4.0]
        assert approx(vector_dot(v, v), 25.0)


class TestVectorNorm:
    def test_euclidean(self):
        assert approx(vector_norm([3.0, 4.0]), 5.0)

    def test_unit(self):
        assert approx(vector_norm([1.0, 0.0]), 1.0)

    def test_zero(self):
        assert vector_norm([0.0, 0.0, 0.0]) == 0.0


class TestMatrixMultiply:
    def test_identity(self):
        I = Matrix([[1.0, 0.0], [0.0, 1.0]])
        A = Matrix([[2.0, 3.0], [4.0, 5.0]])
        result = matrix_multiply(I, A)
        assert mat_approx(result, A)

    def test_known(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0]])
        B = Matrix([[5.0, 6.0], [7.0, 8.0]])
        C = matrix_multiply(A, B)
        assert approx(C.data[0][0], 19.0)
        assert approx(C.data[0][1], 22.0)
        assert approx(C.data[1][0], 43.0)
        assert approx(C.data[1][1], 50.0)

    def test_non_square(self):
        A = Matrix([[1.0, 2.0, 3.0]])
        B = Matrix([[4.0], [5.0], [6.0]])
        C = matrix_multiply(A, B)
        assert approx(C.data[0][0], 32.0)


class TestMatrixTranspose:
    def test_square(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0]])
        T = matrix_transpose(A)
        assert approx(T.data[0][1], 3.0)
        assert approx(T.data[1][0], 2.0)

    def test_rectangular(self):
        A = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        T = matrix_transpose(A)
        assert T.rows == 3
        assert T.cols == 2

    def test_double_transpose(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert mat_approx(matrix_transpose(matrix_transpose(A)), A)


class TestMatrixInverse:
    def test_identity_inverse(self):
        I = Matrix([[1.0, 0.0], [0.0, 1.0]])
        Iinv = matrix_inverse(I)
        assert mat_approx(Iinv, I)

    def test_known_2x2(self):
        A = Matrix([[4.0, 7.0], [2.0, 6.0]])
        Ainv = matrix_inverse(A)
        prod = matrix_multiply(A, Ainv)
        assert approx(prod.data[0][0], 1.0, tol=1e-8)
        assert approx(prod.data[0][1], 0.0, tol=1e-8)
        assert approx(prod.data[1][0], 0.0, tol=1e-8)
        assert approx(prod.data[1][1], 1.0, tol=1e-8)

    def test_3x3(self):
        A = Matrix([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
        Ainv = matrix_inverse(A)
        prod = matrix_multiply(A, Ainv)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert approx(prod.data[i][j], expected, tol=1e-8)


class TestQRDecomposition:
    def test_returns_two_matrices(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = qr_decomposition(A)
        assert len(result) == 2
        Q, R = result
        assert isinstance(Q, Matrix)
        assert isinstance(R, Matrix)

    def test_q_orthonormal_columns(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = qr_decomposition(A)
        Qt = matrix_transpose(Q)
        QtQ = matrix_multiply(Qt, Q)
        n = QtQ.rows
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                assert approx(QtQ.data[i][j], expected, tol=1e-6)

    def test_qr_reconstructs_a(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0]])
        Q, R = qr_decomposition(A)
        reconstructed = matrix_multiply(Q, R)
        assert mat_approx(reconstructed, A, tol=1e-6)
