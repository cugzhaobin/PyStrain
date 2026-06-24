"""
outlier_detection.py — GNSS 速度场异常值检测与剔除

两步策略：
1. 剖分前：KNN 单点预筛 — 基于邻居速度中位数偏差
2. 剖分后：三角网残差 IQR 迭代 — 基于形函数插值预测残差
"""

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# 第一步：剖分前 KNN 预筛
# ---------------------------------------------------------------------------

def knn_prescreening(lon, lat, ve, vn, se, sn, names=None,
                     k_neighbors=8, mad_factor=3.5):
    """
    基于 K 近邻的速度异常预筛

    对于每个站点，对比其速度与 K 个最近邻站点的中位数。
    使用 MAD (Median Absolute Deviation) 作为稳健的离差度量。

    参数
    ----
    lon, lat : 1D arrays
    ve, vn : 1D arrays — 速度分量 (mm/yr)
    se, sn : 1D arrays — 不确定度
    names : 1D array-like
    k_neighbors : int — 最近邻数 (默认 8)
    mad_factor : float — MAD 倍数阈值 (默认 3.5，保守)

    返回
    ----
    outlier_mask : 1D bool array
        True = 被标记为异常
    suspended : 1D bool array
        True = 被标记为悬挂（邻居太少）
    """
    n = len(ve)
    outlier_mask = np.zeros(n, dtype=bool)
    suspended = np.zeros(n, dtype=bool)

    if n < k_neighbors + 2:
        # 站点太少，不检测
        return outlier_mask, suspended

    # 经纬度 → 球面距离的近似平面投影
    # 使用 kd-tree 在经纬度空间中找最近邻（对局部异常检测足够）
    xy = np.column_stack([lon, lat])
    tree = KDTree(xy)

    k_actual = min(k_neighbors + 1, n)  # +1 因为包含自己
    distances, indices = tree.query(xy, k=k_actual)

    for i in range(n):
        if suspended[i]:
            continue

        # 排除自身
        neighbors = indices[i][indices[i] != i]
        if len(neighbors) < 3:
            suspended[i] = True
            continue

        # 邻居速度中位数
        ve_med = np.median(ve[neighbors])
        vn_med = np.median(vn[neighbors])

        # 邻居速度的 MAD
        ve_mad = np.median(np.abs(ve[neighbors] - ve_med))
        vn_mad = np.median(np.abs(vn[neighbors] - vn_med))

        # 避免除以 0
        ve_mad = max(ve_mad, 1e-6)
        vn_mad = max(vn_mad, 1e-6)

        # 偏差（标量度量）
        dev_e = abs(ve[i] - ve_med) / ve_mad
        dev_n = abs(vn[i] - vn_med) / vn_mad
        deviation = np.sqrt(dev_e**2 + dev_n**2)

        if deviation > mad_factor * np.sqrt(2):  # 2D 联合阈值
            outlier_mask[i] = True

    return outlier_mask, suspended


# ---------------------------------------------------------------------------
# 第二步：三角网残差 IQR 迭代
# ---------------------------------------------------------------------------

def compute_residuals_triangulation(tri, xy, good_mask, ve, vn, se, sn):
    """
    计算每个站点在三角网中的预测残差

    使用面积加权平均。站点作为三角形顶点，预测值 =
    去除该站点后其余两个顶点在三角形内的插值（leave-one-out 思想）。

    更简单的做法：统计该站点所属所有三角形的速度变化，
    取该站点速度与邻站速度的一致性作为度量。

    这里用：站点实际速度 vs 邻站的局部均值。

    返回
    ----
    residuals : dict {site_index: {'e_res': ..., 'n_res': ..., 'norm': ...}}
    """
    residuals = {}
    n_sites = len(ve)

    for site_idx in range(n_sites):
        # 查找该站点所属的所有三角形
        neighboring_sites = set()
        for tri_idx in range(tri.simplices.shape[0]):
            if not good_mask[tri_idx]:
                continue
            if site_idx in tri.simplices[tri_idx]:
                for v in tri.simplices[tri_idx]:
                    if v != site_idx:
                        neighboring_sites.add(v)

        if len(neighboring_sites) < 2:
            continue

        neighbors = list(neighboring_sites)

        # 用邻居速度中位数作为预测
        ve_pred = np.median(ve[neighbors])
        vn_pred = np.median(vn[neighbors])

        ve_res = ve[site_idx] - ve_pred
        vn_res = vn[site_idx] - vn_pred
        res_norm = np.sqrt(ve_res**2 + vn_res**2)

        residuals[site_idx] = {
            've_res': ve_res, 'vn_res': vn_res,
            'res_norm': res_norm,
            'n_neighbors': len(neighbors),
        }

    return residuals


def iqr_outlier_detection(residuals, iqr_factor=1.5):
    """
    基于残差模长的 IQR 异常值检测

    参数
    ----
    residuals : dict
    iqr_factor : float — IQR 倍数（默认 1.5）

    返回
    ----
    outlier_sites : list of int
        异常站点索引
    """
    if not residuals:
        return []

    norms = np.array([r['res_norm'] for r in residuals.values()])
    site_indices = list(residuals.keys())

    q1 = np.percentile(norms, 25)
    q3 = np.percentile(norms, 75)
    iqr = q3 - q1

    threshold = q3 + iqr_factor * iqr

    outlier_sites = []
    for idx, norm in zip(site_indices, norms):
        if norm > threshold and norm > 0.5:  # 至少 > 0.5 mm/yr (有意义)
            outlier_sites.append(idx)

    return outlier_sites


# ---------------------------------------------------------------------------
# 完整的迭代异常值剔除
# ---------------------------------------------------------------------------

def iterative_outlier_removal(lon, lat, ve, vn, se, sn, rho, names,
                              triangulation_fn,
                              k_neighbors=8, mad_factor=3.5,
                              iqr_factor=1.5, max_iterations=5):
    """
    完整的异常值检测与剔除流程

    伪代码:
        for iteration in range(max_iterations):
            1. KNN 预筛
            2. 三角剖分
            3. 残差 IQR 检测
            4. 合并异常值，移除
            5. 若本轮无新异常值 → 退出

    参数
    ----
    lon, lat, ..., names : arrays
        站点数据
    triangulation_fn : callable
        函数 sign (lon, lat, ve, vn, keep_mask) → (tri, xy, good_mask)
        注意：需要传入当前 keep_mask
    k_neighbors, mad_factor : KNN 参数
    iqr_factor : IQR 阈值因子
    max_iterations : 最大迭代次数

    返回
    ----
    final_keep_mask : 1D bool array
    outlier_history : list of dict
        每轮被剔除站点的记录
    """
    n = len(ve)
    keep_mask = np.ones(n, dtype=bool)
    outlier_history = []

    for it in range(max_iterations):
        n_current = keep_mask.sum()
        if n_current < 6:
            break  # 站点太少，无法继续

        # 取出当前保留的站点数据
        # 构建全局索引 → 当前子索引的映射
        global_idx = np.where(keep_mask)[0]

        # 1. KNN 预筛（在当前保留站点上做，排除已知异常值）
        outlier_mask_knn, _ = knn_prescreening(
            lon[keep_mask], lat[keep_mask],
            ve[keep_mask], vn[keep_mask],
            se[keep_mask], sn[keep_mask],
            names[keep_mask] if names is not None else None,
            k_neighbors=k_neighbors, mad_factor=mad_factor
        )

        # 2. 三角剖分（用非异常站点）
        sub_keep = ~outlier_mask_knn
        sub_global_idx = global_idx[sub_keep]

        try:
            tri, xy, good_mask = triangulation_fn(
                lon, lat, ve, vn, np.array(sub_global_idx))
        except Exception as e:
            # 剖分失败，记录并退出迭代
            break

        # 3. 残差 IQR（在当前三角网上检测）
        residuals = compute_residuals_triangulation(
            tri, xy, good_mask, ve, vn, se, sn)

        outlier_sites_iqr = iqr_outlier_detection(residuals,
                                                  iqr_factor=iqr_factor)

        # 4. 合并异常值
        # KNN 异常值（在全局索引中）
        knn_global_outliers = set(global_idx[outlier_mask_knn])
        iqr_global_outliers = set(outlier_sites_iqr)

        all_new_outliers = knn_global_outliers | iqr_global_outliers

        # 过滤：只保留目前尚未剔除的
        new_outliers_this_round = [s for s in all_new_outliers if keep_mask[s]]

        if not new_outliers_this_round:
            break

        # 记录
        for s in new_outliers_this_round:
            reason = []
            if s in knn_global_outliers:
                reason.append('KNN')
            if s in iqr_global_outliers:
                # 获取残差
                res = residuals.get(s, {})
                reason.append(f"IQR(res={res.get('res_norm', 0):.2f})")
            outlier_history.append({
                'name': names[s] if names is not None else f'site_{s}',
                'lon': lon[s],
                'lat': lat[s],
                'residual': residuals.get(s, {}).get('res_norm', 0.0),
                'reason': '+'.join(reason),
                'iteration': it,
            })

        # 更新 keep_mask
        for s in new_outliers_this_round:
            keep_mask[s] = False

    return keep_mask, outlier_history
