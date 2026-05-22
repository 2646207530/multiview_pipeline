import numpy as np
import torch


# coordinate transformation matrix
euler_coordtrans_RIGHT = np.array([
       [   0,  -6,  10,
           0,   1,  -5,
           0, -10,   0,
           0,  10,   2,
           0,  10,  -2,
           0,  10,  10,
          30,  35,  10,
          20,  42,   8,
          15,  45,   8,
           0,  12,   5,
           0,  25,  -5,
           0,  25,  15,
          30, -75, -35,
          20, -45, -30,
          20, -45, -30]]) / 180 * np.pi


def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    return rotMat


def coordtrans(rotmat, local2global, transtype):

        batchsize = rotmat.shape[0]
        local2global = local2global.clone()
        if transtype == 1:
            rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize])
            rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize].transpose(2,3), rotmat[:,1:])
        else:
            rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize].transpose(2,3))
            rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize], rotmat[:,1:])

        return rotmat


def batch_compute_similarity_transform_numpy(S1, S2, R_GT=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2, axis=1).sum(axis=1)

    # 3. The outer product of X1 and X2.
    K = np.matmul(X1, X2.transpose(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.transpose(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[1])[None]
    Z = Z.repeat(U.shape[0], axis=0)
    Z[:, -1, -1] *= np.sign(np.linalg.det(np.matmul(U, V.transpose(0, 2, 1))))

    # Construct R.
    if R_GT is None:
        R = np.matmul(V, np.matmul(Z, U.transpose(0, 2, 1)))
    else:
        R = R_GT

    # 5. Recover scale.
    scale = np.concatenate([np.trace(x)[None] for x in np.matmul(R, K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale[:, None, None] * (np.matmul(R, mu1)))

    # 7. Error:
    S1_hat = scale[:, None, None] * np.matmul(R, S1) + t

    return S1_hat, (scale, R, t)

def batch_compute_similarity_transform_torch(S1, S2, R_GT=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (B x 3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    '''
    # 1. Remove mean.
    mu1 = S1.mean(dim=-1, keepdim=True)
    mu2 = S2.mean(dim=-1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    
    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)  # (B,)
    
    # 3. The outer product of X1 and X2.
    K = X1 @ X2.permute(0, 2, 1)  # (B, 3, 3)
    
    # Construct R.
    if R_GT is None:
        # 4. SVD solution to maximize trace(R'K)
        U, s, V = torch.svd(K)
        
        # Compute M = U @ V^T for each batch
        M = U @ V.permute(0, 2, 1)
        
        # Batch-compute determinants for 3x3 matrices
        det = (
            M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1]) -
            M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) +
            M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
        )
        
        # Construct correction matrix Z
        Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] = torch.sign(det)  # Ensure det(R)=1
        
        R = V @ (Z @ U.permute(0, 2, 1))
    else:
        R = R_GT
    
    # 5. Recover scale (batch trace without loop)
    trace = torch.diagonal(R @ K, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
    scale = trace / var1  # (B,)
    
    # 6. Recover translation
    t = mu2 - scale.view(-1, 1, 1) * (R @ mu1)
    
    # 7. Transform S1
    S1_hat = scale.view(-1, 1, 1) * (R @ S1) + t
    
    return S1_hat, (scale, R, t)

def batch_remove_align(coords, scale, R, t):
    R = R.permute(0, 2, 1)
    coords = coords.permute(0, 2, 1)
    res = (torch.matmul(R, coords) - t) / scale[:, None, None]
    return res.permute(0, 2, 1)

