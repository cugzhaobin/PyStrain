"""
uncertainty.py — 应变率不确定性计算 (Monte Carlo)

利用 GNSS 速度场不确定度 (Se, Sn, Rho) 传播到应变率输出。
固定三角网拓扑，重采样速度场 M 次，统计应变率分量标准差。
"""

import numpy as np
from triangulation import compute_shape_function_derivatives
from strain_rate import (compute_velocity_gradient, strain_from_gradient,
                         principal_strain, strain_invariants)


def monte_carlo_uncertainty(tri, xy, good_mask, ve, vn, se, sn, rho,
                            n_iterations=500, seed=42):
    """
    Monte Carlo 传播速度场不确定度到应变率

    固定三角网拓扑。每次迭代：
    1. 从 N(0, Σ_i) 抽取各站点速度扰动
    2. 对扰动速度场计算全部三角形应变率
    3. 保存结果

    参数
    ----
    tri : Delaunay
    xy : (N, 2) 投影坐标
    good_mask : (N_tri,) bool
    ve, vn : (N,) 站点速度 (mm/yr)
    se, sn : (N,) 速度不确定度 (mm/yr)
    rho : (N,) 相关系数
    n_iterations : int
        Monte Carlo 迭代次数 (默认 500)
    seed : int
        随机数种子

    返回
    ----
    uncertainty : dict
        **key_std: 标准差
        **key_mean: Monte Carlo 均值
    """
    rng = np.random.default_rng(seed)
    n_sites = len(ve)

    # 预计算形函数
    B_list, good_indices, areas = compute_shape_function_derivatives(
        tri, xy, good_mask)
    n_good = len(good_indices)

    if n_good == 0:
        raise RuntimeError("No valid triangles for MC uncertainty!")

    # 预构建各站点协方差矩阵的 Cholesky 分解
    # Σ_i = [[Se², ρ*Se*Sn], [ρ*Se*Sn, Sn²]]
    # 抽取: L @ N(0,1) 其中 L = Cholesky(Σ)
    L_list = []
    for i in range(n_sites):
        Si = np.array([[se[i]**2, rho[i] * se[i] * sn[i]],
                       [rho[i] * se[i] * sn[i], sn[i]**2]])
        # 用 eigenvalue 修正保证正定
        eigvals, eigvecs = np.linalg.eigh(Si)
        eigvals = np.maximum(eigvals, 1e-30)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        L_list.append(L)

    # 存储所有迭代结果
    all_e_ee = np.zeros((n_iterations, n_good))
    all_e_en = np.zeros((n_iterations, n_good))
    all_e_nn = np.zeros((n_iterations, n_good))
    all_e1 = np.zeros((n_iterations, n_good))
    all_e2 = np.zeros((n_iterations, n_good))
    all_dilatation = np.zeros((n_iterations, n_good))
    all_max_shear = np.zeros((n_iterations, n_good))

    for it in range(n_iterations):
        # 生成扰动速度
        ve_pert = np.zeros(n_sites)
        vn_pert = np.zeros(n_sites)

        for i in range(n_sites):
            z = rng.normal(0, 1, 2)
            delta = L_list[i] @ z
            ve_pert[i] = ve[i] + delta[0]
            vn_pert[i] = vn[i] + delta[1]

        # 计算应变率
        for k, (idx, B) in enumerate(zip(good_indices, B_list)):
            verts = tri.simplices[idx]
            ve_tri = ve_pert[verts]
            vn_tri = vn_pert[verts]

            L = compute_velocity_gradient(B, ve_tri, vn_tri)
            ee, en, enn, om = strain_from_gradient(L)
            p1, p2, azi = principal_strain(ee, en, enn)
            dil, ms, si = strain_invariants(p1, p2)

            all_e_ee[it, k] = ee
            all_e_en[it, k] = en
            all_e_nn[it, k] = enn
            all_e1[it, k] = p1
            all_e2[it, k] = p2
            all_dilatation[it, k] = dil
            all_max_shear[it, k] = ms

    # 单位转换: mm/(km·yr) → nstrain/yr (×1000)
    UNIT_CONV = 1000.0

    e_ee_mean = np.mean(all_e_ee, axis=0) * UNIT_CONV
    e_en_mean = np.mean(all_e_en, axis=0) * UNIT_CONV
    e_nn_mean = np.mean(all_e_nn, axis=0) * UNIT_CONV
    e1_mean = np.mean(all_e1, axis=0) * UNIT_CONV
    e2_mean = np.mean(all_e2, axis=0) * UNIT_CONV
    dilatation_mean = np.mean(all_dilatation, axis=0) * UNIT_CONV
    max_shear_mean = np.mean(all_max_shear, axis=0) * UNIT_CONV

    e_ee_std = np.std(all_e_ee, axis=0, ddof=1) * UNIT_CONV
    e_en_std = np.std(all_e_en, axis=0, ddof=1) * UNIT_CONV
    e_nn_std = np.std(all_e_nn, axis=0, ddof=1) * UNIT_CONV
    e1_std = np.std(all_e1, axis=0, ddof=1) * UNIT_CONV
    e2_std = np.std(all_e2, axis=0, ddof=1) * UNIT_CONV
    dilatation_std = np.std(all_dilatation, axis=0, ddof=1) * UNIT_CONV
    max_shear_std = np.std(all_max_shear, axis=0, ddof=1) * UNIT_CONV

    # 原始值也转换（调试用）
    all_e_ee *= UNIT_CONV
    all_e_en *= UNIT_CONV
    all_e_nn *= UNIT_CONV
    all_e1 *= UNIT_CONV
    all_e2 *= UNIT_CONV
    all_dilatation *= UNIT_CONV
    all_max_shear *= UNIT_CONV

    return {
        'e_ee_mean': e_ee_mean, 'e_en_mean': e_en_mean, 'e_nn_mean': e_nn_mean,
        'e1_mean': e1_mean, 'e2_mean': e2_mean,
        'dilatation_mean': dilatation_mean,
        'max_shear_mean': max_shear_mean,

        'e_ee_std': e_ee_std, 'e_en_std': e_en_std, 'e_nn_std': e_nn_std,
        'e1_std': e1_std, 'e2_std': e2_std,
        'dilatation_std': dilatation_std,
        'max_shear_std': max_shear_std,

        'all_e_ee': all_e_ee, 'all_e_en': all_e_en, 'all_e_nn': all_e_nn,
        'all_e1': all_e1, 'all_e2': all_e2,
        'all_dilatation': all_dilatation,
        'all_max_shear': all_max_shear,
    }
