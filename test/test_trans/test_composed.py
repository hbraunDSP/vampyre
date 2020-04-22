"""
test_fft2.py: test suite for the fft2 transform module
"""

from __future__ import print_function, division

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def fft2_round_trip_is_consistent(nrow=256,ncol=256, err_tol = 1E-12):
    fft_size = (nrow, ncol)
    raise NotImplementedError
    
class TestCases(unittest.TestCase):
    def test_2_random_matrices_composition(self):
        """
        Compare the forward and adjoint transform operations for 
        composition of 2 random matrices.
        """

        mat_size = 256 #using all square matrices
        rtol= 0
        atol = 1E-12

        A = vp.trans.rand_rot_invariant_mat(nz1=mat_size, nz0=mat_size)
        B = vp.trans.rand_rot_invariant_mat(nz1=mat_size, nz0=mat_size)
        A_H = np.conjugate(np.transpose(A))
        B_H = np.conjugate(np.transpose(B))
        x = np.random.randn(mat_size)
        y_expected = np.matmul(B, np.matmul(A, x))
        x_expected = np.matmul(A_H, np.matmul(B_H, y_expected))

        composed_op = vp.trans.ComposedLT(
            [vp.trans.MatrixLT(A,[mat_size]), vp.trans.MatrixLT(B, [mat_size])])
        y = composed_op.dot(x)
        x_roundtrip = composed_op.dotH(y)
        
        self.assertTrue(np.allclose(y, y_expected, rtol, atol))
        self.assertTrue(np.allclose(x_roundtrip, x_expected, rtol, atol))

if __name__ == '__main__':    
    unittest.main()