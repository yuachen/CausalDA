import torch

def _mix_rbf_kernel(X, Y, sigma_list):
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def mix_rbf_mmd2(X, Y, sigma_list):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY):
    m = K_XX.size(0)
    l = K_YY.size(0)

    K_XX_sum = K_XX.sum()
    K_YY_sum = K_YY.sum()
    K_XY_sum = K_XY.sum()

    mmd2 = (K_XX_sum / (m * m) 
        + K_YY_sum / (l * l) 
        - 2.0 * K_XY_sum / (m * l))
        
    return mmd2
