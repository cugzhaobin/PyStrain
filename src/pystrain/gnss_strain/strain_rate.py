"""
strain_rate.py — GNSS 速度场 → 应变率张量 核心计算

从三角剖分 + 形函数计算每个三角形的应变率。
包含光滑约束的后处理加权平均。
"""

import numpy as np
from triangulation import (compute_shape_function_derivatives,
                           build_adjacency, triangle_centroids_ll,
                           detect_hanging_sites, _triangle_area)


# ---------------------------------------------------------------------------
# 速度梯度 → 应变率（单三角形）
# ---------------------------------------------------------------------------

def compute_velocity_gradient(B, ve_tri, vn_tri):
    """
    由形函数导数和三角形三顶点速度计算速度梯度张量

    参数
    ----
    B : (2, 3) array
        形函数导数矩阵 [∂N/∂x; ∂N/∂y]
    ve_tri, vn_tri : (3,) array
        三角形三站点的东向/北向速度

    返回
    ----
    L : (2, 2) array
        速度梯度张量 [[dve/dx, dve/dy], [dvn/dx, dvn/dy]]
    """
    L = np.zeros((2, 2))
    # L[0,:] = [dve/dx, dve/dy]  = B @ ve
    L[0, 0] = np.dot(B[0, :], ve_tri)
    L[0, 1] = np.dot(B[1, :], ve_tri)
    # L[1,:] = [dvn/dx, dvn/dy]  = B @ vn
    L[1, 0] = np.dot(B[0, :], vn_tri)
    L[1, 1] = np.dot(B[1, :], vn_tri)
    return L


def strain_from_gradient(L):
    """
    速度梯度张量 → 应变率张量 + 旋转率

    返回
    ----
    e_ee, e_en, e_nn : 应变率分量 (10^-9 / yr = nstrain/yr)
    omega : 旋转率
    """
    e_ee = L[0, 0]
    e_en = 0.5 * (L[0, 1] + L[1, 0])
    e_nn = L[1, 1]
    omega = 0.5 * (L[1, 0] - L[0, 1])
    return e_ee, e_en, e_nn, omega


def principal_strain(e_ee, e_en, e_nn):
    """
    主应变率 + 方向

    特征值分解:
        | e_ee  e_en | = R^T diag(e1, e2) R
        | e_en  e_nn |

    约定: e1 为更张性的主应变（代数上更大）, e2 为更压缩的
          |e1| >= |e2| 不一定成立，但 e1 >= e2 总成立。

    返回
    ----
    e1, e2 : 主应变率
    azimuth : e1 方向（从东方向逆时针，°）
    """
    # 2x2 矩阵特征值
    trace = e_ee + e_nn
    det = e_ee * e_nn - e_en * e_en
    D = np.sqrt(max(trace**2 - 4.0 * det, 0.0))
    e1 = (trace + D) / 2.0  # 更大
    e2 = (trace - D) / 2.0  # 更小

    # e1 特征向量方向
    # 2e_en * v_x + (e_nn - e_ee - D) * v_y = 0
    # 只要 e_en != 0, 特征向量 ∝ (2*e_en, e_ee - e_nn + D)
    if abs(e_en) > 1e-30:
        vx = 2.0 * e_en
        vy = e_ee - e_nn + D
    elif abs(e_ee - e2) > abs(e_nn - e2):
        vx = 1.0
        vy = 0.0
    else:
        vx = 0.0
        vy = 1.0

    # 方向角（从东逆时针）
    azimuth = np.degrees(np.arctan2(vy, vx))
    if azimuth < 0:
        azimuth += 180.0
    if azimuth >= 180.0:
        azimuth -= 180.0

    return e1, e2, azimuth


def strain_invariants(e1, e2):
    """
    应变率不变量

    返回
    ----
    dilatation : 面膨胀率 (div v)
    max_shear : 最大剪应变率
    second_invariant : 第二不变量
    """
    dilatation = e1 + e2
    max_shear = (e1 - e2) / 2.0
    second_invariant = np.sqrt(e1**2 + e2**2)
    return dilatation, max_shear, second_invariant


# ---------------------------------------------------------------------------
# 批量应变率计算
# ---------------------------------------------------------------------------

def compute_strain_for_all_triangles(tri, xy, B_list, good_indices,
                                     ve, vn):
    """
    对全部合格三角形批量计算应变率

    参数
    ----
    tri : Delaunay
    xy : (N, 2) 投影坐标
    B_list : list of (2,3) arrays
    good_indices : 1D int array
    ve, vn : (N,) 1D arrays, 站点速度

    返回
    ----
    result : dict
    """
    n_good = len(good_indices)
    e_ee = np.zeros(n_good)
    e_en = np.zeros(n_good)
    e_nn = np.zeros(n_good)
    omega = np.zeros(n_good)
    e1 = np.zeros(n_good)
    e2 = np.zeros(n_good)
    azimuth = np.zeros(n_good)
    dilatation = np.zeros(n_good)
    max_shear = np.zeros(n_good)
    second_invariant = np.zeros(n_good)

    for k, (idx, B) in enumerate(zip(good_indices, B_list)):
        verts = tri.simplices[idx]
        ve_tri = ve[verts]
        vn_tri = vn[verts]

        L = compute_velocity_gradient(B, ve_tri, vn_tri)
        ee, en, enn, om = strain_from_gradient(L)
        p1, p2, azi = principal_strain(ee, en, enn)
        dil, ms, si = strain_invariants(p1, p2)

        e_ee[k] = ee
        e_en[k] = en
        e_nn[k] = enn
        omega[k] = om
        e1[k] = p1
        e2[k] = p2
        azimuth[k] = azi
        dilatation[k] = dil
        max_shear[k] = ms
        second_invariant[k] = si

    # Unit conversion: xy in km, v in mm/yr
    # Gradient = mm/(km·yr) = 10⁻⁶ /yr = 1000 × 10⁻⁹ /yr = 1000 nstrain/yr
    UNIT_CONV = 1000.0

    # 转换到 nstrain/yr (10⁻⁹/yr)
    e_ee *= UNIT_CONV
    e_en *= UNIT_CONV
    e_nn *= UNIT_CONV
    omega *= UNIT_CONV
    e1 *= UNIT_CONV
    e2 *= UNIT_CONV
    dilatation *= UNIT_CONV
    max_shear *= UNIT_CONV
    second_invariant *= UNIT_CONV

    return {
        'e_ee': e_ee, 'e_en': e_en, 'e_nn': e_nn,
        'omega': omega,
        'e1': e1, 'e2': e2, 'azimuth': azimuth,
        'dilatation': dilatation, 'max_shear': max_shear,
        'second_invariant': second_invariant,
        'tri_indices': good_indices,
    }


# ---------------------------------------------------------------------------
# 光滑约束（后处理加权平均）
# ---------------------------------------------------------------------------

def smooth_strain_rate(result, adjacency, weight=0.3, n_iterations=3):
    """
    三角网空间平滑

    加权平均：ε'_i = (1-w)*ε_i + w*mean(ε_neighbors)

    参数
    ----
    result : dict
        应变率结果（来自 compute_strain_for_all_triangles）
    adjacency : dict
        邻接关系（来自 build_adjacency）
    weight : float
        光滑权重 (0=无平滑, 1=完全平均到邻居)
    n_iterations : int
        迭代次数

    返回
    ----
    result_smooth : dict
        平滑后的应变率（原地修改的副本）
    """
    import copy
    result_smooth = copy.deepcopy(result)

    # 需要平滑的标量场
    fields = ['e_ee', 'e_en', 'e_nn', 'e1', 'e2',
              'dilatation', 'max_shear', 'second_invariant']

    tri_indices = result_smooth['tri_indices']
    idx_to_pos = {idx: k for k, idx in enumerate(tri_indices)}

    for _ in range(n_iterations):
        for field in fields:
            original = result_smooth[field].copy()
            for k, tri_idx in enumerate(tri_indices):
                neighbors = adjacency.get(tri_idx, set())
                # 仅考虑有结果的邻居
                neighbor_positions = [idx_to_pos[n] for n in neighbors
                                      if n in idx_to_pos]
                if not neighbor_positions:
                    continue

                # 边界三角形（邻接少）减小权重
                w = weight
                if len(neighbor_positions) < 2:
                    w = weight * 0.5

                neighbor_mean = np.mean(original[neighbor_positions])
                result_smooth[field][k] = (1.0 - w) * original[k] + w * neighbor_mean

    # 重新计算导出量
    for k in range(len(tri_indices)):
        ee = result_smooth['e_ee'][k]
        en = result_smooth['e_en'][k]
        enn = result_smooth['e_nn'][k]
        p1, p2, azi = principal_strain(ee, en, enn)
        dil, ms, si = strain_invariants(p1, p2)

        result_smooth['e1'][k] = p1
        result_smooth['e2'][k] = p2
        result_smooth['azimuth'][k] = azi
        result_smooth['dilatation'][k] = dil
        result_smooth['max_shear'][k] = ms
        result_smooth['second_invariant'][k] = si

    return result_smooth


# ---------------------------------------------------------------------------
# 三顶点插值：用形函数从三角形应变率插值到站点位置
# ---------------------------------------------------------------------------

def interpolate_to_site(tri, xy, good_mask, result, site_idx):
    """
    将三角形应变率插值回某个站点

    找到包含该站点的所有三角形，面积加权平均
    """
    fields = ['e_ee', 'e_en', 'e_nn', 'e1', 'e2', 'dilatation', 'max_shear']
    field_vals = {f: [] for f in fields}
    weights = []

    good_indices = np.where(good_mask)[0]
    for tri_idx in good_indices:
        if site_idx in tri.simplices[tri_idx]:
            area = _triangle_area(tri, xy, tri_idx)
            weights.append(area)
            pos = np.where(good_indices == tri_idx)[0][0]
            for f in fields:
                field_vals[f].append(result[f][pos])

    if not weights:
        return None

    weights = np.array(weights) / sum(weights)
    interpolated = {}
    for f in fields:
        interpolated[f] = np.sum(np.array(field_vals[f]) * weights)

    # 计算残差相关量
    interpolated['ve_pred'] = 0.0  # 需用形函数计算
    interpolated['vn_pred'] = 0.0
    return interpolated


def compute_site_residuals(tri, xy, good_mask, result, ve, vn):
    """
    计算每个站点在三角网中的预测值 vs 观测值残差

    使用形函数直接在每个站点所在三角形内插值

    返回
    ----
    residuals : dict
        {site_idx: {'ve_res': ..., 'vn_res': ..., 'res_norm': ...}, ...}
    """
    residuals = {}
    B_list, good_indices, _ = compute_shape_function_derivatives(tri, xy, good_mask)

    for site_idx in range(len(ve)):
        # 查找包含该站点的三角形
        ve_pred = []
        vn_pred = []
        weights = []

        for k, tri_idx in enumerate(good_indices):
            if site_idx in tri.simplices[tri_idx]:
                # 计算该站点在三角形内的重心坐标
                verts = tri.simplices[tri_idx]
                p = xy[verts]
                x = xy[site_idx]

                # 面积坐标 = 重心坐标
                area_total = abs(np.cross(p[1] - p[0], p[2] - p[0]))
                if area_total < 1e-30:
                    continue

                a1 = abs(np.cross(p[2] - x, p[1] - x)) / area_total  # 顶点0的权重
                a2 = abs(np.cross(p[0] - x, p[2] - x)) / area_total  # 顶点1的权重
                a3 = abs(np.cross(p[1] - x, p[0] - x)) / area_total  # 顶点2的权重

                # 用形函数插值
                # v_pred(x) = a1*v(顶点0) + a2*v(顶点1) + a3*v(顶点2)
                # 注意：对于不在三角形内部的点，面积坐标和可能>1
                # 我们在这里不判断，因为三角网中站点必然是三角形顶点
                # 但需要站点是三角形顶点！
                v_pred_e = (a1 * ve[verts[0]] + a2 * ve[verts[1]] +
                            a3 * ve[verts[2]])
                v_pred_n = (a1 * vn[verts[0]] + a2 * vn[verts[1]] +
                            a3 * vn[verts[2]])

                ve_pred.append(v_pred_e)
                vn_pred.append(v_pred_n)
                weights.append(1.0)

        if not ve_pred:
            continue

        ve_pred = np.average(ve_pred, weights=weights)
        vn_pred = np.average(vn_pred, weights=weights)

        ve_res = ve[site_idx] - ve_pred
        vn_res = vn[site_idx] - vn_pred
        res_norm = np.sqrt(ve_res**2 + vn_res**2)

        residuals[site_idx] = {
            've_pred': ve_pred, 'vn_pred': vn_pred,
            've_res': ve_res, 'vn_res': vn_res,
            'res_norm': res_norm,
        }

    return residuals


# ---------------------------------------------------------------------------
# 完整流程包装
# ---------------------------------------------------------------------------

def run_strain_rate_pipeline(tri, xy, good_mask, ve, vn, lon, lat,
                             smooth_weight=0.3, smooth_iter=3):
    """
    完整的应变率计算流程

    1. 计算形函数导数
    2. 批量计算应变率
    3. 光滑约束
    4. 组装输出

    参数
    ----
    tri : Delaunay
    xy : (N,2) 投影坐标
    good_mask : (N_tri,) bool
    ve, vn : (N,) 速度 (mm/yr)
    lon, lat : (N,) 经纬度
    smooth_weight : float
    smooth_iter : int

    返回
    ----
    result : dict (应变率结果)
    adjacency : dict (邻接关系)
    centroids_lon, centroids_lat : 1D arrays
    """
    # 1. 形函数
    B_list, good_indices, areas = compute_shape_function_derivatives(
        tri, xy, good_mask)

    if len(good_indices) == 0:
        raise RuntimeError("No valid triangles after quality filtering!")

    # 2. 应变率
    result = compute_strain_for_all_triangles(
        tri, xy, B_list, good_indices, ve, vn)
    result['areas'] = areas
    result['tri_ids'] = good_indices

    # 3. 邻接 + 光滑
    adjacency = build_adjacency(tri, good_mask)

    result = smooth_strain_rate(result, adjacency,
                                weight=smooth_weight,
                                n_iterations=smooth_iter)

    # 4. 重心坐标
    centroids_lon, centroids_lat = triangle_centroids_ll(
        tri, lon, lat, good_mask)

    result['centroids_lon'] = centroids_lon
    result['centroids_lat'] = centroids_lat

    return result, adjacency
