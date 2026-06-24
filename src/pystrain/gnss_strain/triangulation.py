"""
triangulation.py — Delaunay 三角剖分 + 形函数 + 网格质量控制

核心功能:
1. 坐标投影 (WGS84 → UTM / 等距方位投影)
2. Delaunay 三角剖分
3. 多边形边界裁剪
4. 网格质量控制（边长、最小角、面积筛选）
5. 形函数导数矩阵计算
6. 三角网邻接关系构建
"""

import numpy as np
from scipy.spatial import Delaunay, KDTree
from matplotlib.path import Path


# ---------------------------------------------------------------------------
# 坐标投影
# ---------------------------------------------------------------------------

def _ll_to_utm(lon, lat):
    """
    将经纬度投影到 UTM 平面坐标

    使用 WGS84 椭球参数。区域较小（< 6° 经度宽）时误差可忽略。

    参数
    ----
    lon, lat : array-like, 度

    返回
    ----
    x, y : array-like, 米
    proj_params : dict — 投影参数，用于逆变换
    """
    a = 6378137.0            # 半长轴 (m)
    e2 = 0.00669437999014    # 第一偏心率平方

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # 中心点（用于 UTM 条带中心经度计算）
    lon0 = (np.floor((np.mean(lon) + 180.0) / 6.0) * 6.0 - 180.0) + 3.0

    # 转换到弧度
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    lon0_r = np.radians(lon0)

    # UTM 投影公式
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    tan_lat = np.tan(lat_r)

    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    T = tan_lat**2
    C = e2 / (1.0 - e2) * cos_lat**2
    A_ = (lon_r - lon0_r) * cos_lat

    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0) * lat_r
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0) * np.sin(2.0 * lat_r)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0) * np.sin(4.0 * lat_r)
        - (35.0 * e2**3 / 3072.0) * np.sin(6.0 * lat_r)
    )

    k0 = 0.9996
    x = k0 * N * (A_ + (1.0 - T + C) * A_**3 / 6.0
                   + (5.0 - 18.0 * T + T**2 + 72.0 * C - 58.0 * e2) * A_**5 / 120.0)
    y = k0 * (M + N * tan_lat * (
        A_**2 / 2.0 + (5.0 - T + 9.0 * C + 4.0 * C**2) * A_**4 / 24.0
        + (61.0 - 58.0 * T + T**2 + 600.0 * C - 330.0 * e2) * A_**6 / 720.0
    ))

    proj_params = {'lon0': lon0, 'a': a, 'e2': e2, 'k0': k0}
    return x, y, proj_params


def _utm_to_ll(x, y, proj_params):
    """UTM → 经纬度逆变换（简化版，非关键路径，暂不需要）"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 三角剖分与质量控制
# ---------------------------------------------------------------------------

def delaunay_triangulation(lon, lat, polygon=None, min_angle_deg=10.0,
                           max_edge_pctl=95.0, max_edge_factor=1.5,
                           min_area_ratio=0.1, max_edge_km=None):
    """
    对 GNSS 站点进行 Delaunay 三角剖分，施加质量控制

    参数
    ----
    lon, lat : 1D array
        站点经纬度（度）
    polygon : (N,2) array or None
        研究区边界多边形 (lon, lat)。None 则用凸包。
    min_angle_deg : float
        三角形最小内角阈值（度），低于此值剔除
    max_edge_pctl : float
        最大边长百分位阈值
    max_edge_factor : float
        最大边长 = pctl(站点间距, max_edge_pctl) * max_edge_factor
    min_area_ratio : float
        最小面积 = p05(三角形面积) * min_area_ratio
    max_edge_km : float or None
        三角形边长绝对上限（km）；超过此值的三角形直接剔除。
        None 表示不启用。与 max_edge_pctl/factor 独立叠加。

    返回
    ----
    tri : Delaunay 对象
    good_triangles : 1D bool array
        标记哪些三角形合格（在保留集中）
    xy : (N, 2) array
        投影坐标（米）
    proj_params : dict
        投影参数
    """
    # 1. 投影 (转换为 km 坐标，使应变率自然单位 = 10⁻⁶/yr)
    x, y, proj_params = _ll_to_utm(lon, lat)
    xy = np.column_stack([x / 1000.0, y / 1000.0])  # m → km

    # 2. Delaunay 全剖分
    tri = Delaunay(xy)

    n_tri = tri.simplices.shape[0]
    good = np.ones(n_tri, dtype=bool)

    # 3. 多边形裁剪
    if polygon is not None:
        _apply_polygon_mask(tri, xy, lon, lat, polygon, good)

    # 4. 三连筛
    _filter_large_edges(tri, xy, good, max_edge_pctl, max_edge_factor)
    _filter_small_angles(tri, xy, good, min_angle_deg)
    _filter_small_area(tri, xy, good, min_area_ratio)

    # 5. 绝对边长上限（可选）
    if max_edge_km is not None:
        _filter_abs_edge(tri, xy, good, max_edge_km)

    return tri, good, xy, proj_params


def _apply_polygon_mask(tri, xy, lon, lat, polygon, good):
    """
    仅保留重心在多边形内部的三角形

    多边形坐标 (lon, lat) → 先转换到 UTM，再构建 Path
    """
    # 转换多边形到 UTM (然后转 km 以匹配 xy)
    if polygon.shape[1] >= 2:
        px, py, _ = _ll_to_utm(polygon[:, 0], polygon[:, 1])
        poly_xy = np.column_stack([px / 1000.0, py / 1000.0])
    else:
        poly_xy = polygon

    path = Path(poly_xy)

    centroids = _triangle_centroids(tri, xy)
    inside = path.contains_points(centroids)

    good[~inside] = False


def _filter_large_edges(tri, xy, good, pctl=95.0, factor=1.5):
    """剔除包含过长边的三角形"""
    if not good.any():
        return

    # 计算所有站点间距离的分位数（用 KDTree 采样边界站点间的距离）
    all_distances = _compute_site_distances(xy)
    threshold = np.percentile(all_distances, pctl) * factor

    for i in range(tri.simplices.shape[0]):
        if not good[i]:
            continue
        verts = tri.simplices[i]
        p = xy[verts]
        d1 = np.linalg.norm(p[1] - p[0])
        d2 = np.linalg.norm(p[2] - p[1])
        d3 = np.linalg.norm(p[0] - p[2])
        if max(d1, d2, d3) > threshold:
            good[i] = False


def _filter_small_angles(tri, xy, good, min_angle_deg=10.0):
    """剔除包含过小内角的三角形"""
    if not good.any():
        return

    min_angle_rad = np.radians(min_angle_deg)

    for i in range(tri.simplices.shape[0]):
        if not good[i]:
            continue
        verts = tri.simplices[i]
        p = xy[verts]
        # 三边
        a = np.linalg.norm(p[1] - p[2])
        b = np.linalg.norm(p[2] - p[0])
        c = np.linalg.norm(p[0] - p[1])
        # 用余弦定理计算三个内角
        angles = []
        # 顶点0对应的角
        cos_a = (b**2 + c**2 - a**2) / (2.0 * b * c + 1e-30)
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        # 顶点1对应的角
        cos_b = (a**2 + c**2 - b**2) / (2.0 * a * c + 1e-30)
        angles.append(np.arccos(np.clip(cos_b, -1.0, 1.0)))
        # 顶点2对应的角
        cos_c = (a**2 + b**2 - c**2) / (2.0 * a * b + 1e-30)
        angles.append(np.arccos(np.clip(cos_c, -1.0, 1.0)))

        if min(angles) < min_angle_rad:
            good[i] = False


def _filter_small_area(tri, xy, good, min_area_ratio=0.1):
    """剔除面积过小的退化三角形"""
    if not good.any():
        return

    # 仅用当前合格的三角形统计面积分位数
    good_indices = np.where(good)[0]
    areas = np.array([_triangle_area(tri, xy, i) for i in good_indices])
    if len(areas) < 3:
        return

    threshold = np.percentile(areas, 5.0) * min_area_ratio

    for i in range(tri.simplices.shape[0]):
        if not good[i]:
            continue
        if _triangle_area(tri, xy, i) < threshold:
            good[i] = False


def _filter_abs_edge(tri, xy, good, max_edge_km):
    """剔除含有超过绝对长度上限的边的三角形（xy 单位为 km）"""
    if not good.any():
        return
    for i in range(tri.simplices.shape[0]):
        if not good[i]:
            continue
        verts = tri.simplices[i]
        p = xy[verts]
        d1 = np.linalg.norm(p[1] - p[0])
        d2 = np.linalg.norm(p[2] - p[1])
        d3 = np.linalg.norm(p[0] - p[2])
        if max(d1, d2, d3) > max_edge_km:
            good[i] = False


def _compute_site_distances(xy):
    """
    高效计算站点间距离样本（用于百分位数估计）

    随机抽样 5000 对站点计算距离
    """
    n = len(xy)
    if n <= 500:
        # 全组合
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(xy[i] - xy[j]))
        return np.array(dists)
    else:
        # 随机抽样
        np.random.seed(42)
        n_sample = 5000
        dists = np.zeros(n_sample)
        for k in range(n_sample):
            i, j = np.random.choice(n, 2, replace=False)
            dists[k] = np.linalg.norm(xy[i] - xy[j])
        return dists


# ---------------------------------------------------------------------------
# 几何工具函数
# ---------------------------------------------------------------------------

def _triangle_area(tri, xy, idx):
    """计算三角形面积（m²，投影坐标）"""
    verts = tri.simplices[idx]
    p = xy[verts]
    return 0.5 * abs(np.cross(p[1] - p[0], p[2] - p[0]))


def _triangle_centroids(tri, xy):
    """计算所有三角形的重心"""
    centroids = np.mean(xy[tri.simplices], axis=1)
    return centroids


def triangle_centroids_ll(tri, lon, lat, good_mask):
    """计算合格三角形的重心（经纬度）"""
    centroids_lon = np.mean(lon[tri.simplices], axis=1)
    centroids_lat = np.mean(lat[tri.simplices], axis=1)
    return centroids_lon[good_mask], centroids_lat[good_mask]


# ---------------------------------------------------------------------------
# 形函数（Shape Functions）
# ---------------------------------------------------------------------------

def compute_shape_function_derivatives(tri, xy, good_mask):
    """
    对每个（合格）三角形计算形函数导数矩阵 B

    线性三角单元：
        N_i(x,y) = a_i + b_i*x + c_i*y

        ∂N1/∂x = (y2-y3)/(2A),  ∂N1/∂y = (x3-x2)/(2A)
        ∂N2/∂x = (y3-y1)/(2A),  ∂N2/∂y = (x1-x3)/(2A)
        ∂N3/∂x = (y1-y2)/(2A),  ∂N3/∂y = (x2-x1)/(2A)

    参数
    ----
    tri : Delaunay
    xy : (N, 2) array, 投影坐标
    good_mask : (N_tri,) bool array

    返回
    ----
    B_list : list of (2,3) arrays
        每个三角形的 ∂(v)/∂(x) 算子矩阵
    good_indices : 1D int array
        合格三角形的序号
    areas : 1D array
        三角形面积 (m²)
    """
    good_indices = np.where(good_mask)[0]
    B_list = []
    areas = []

    for idx in good_indices:
        verts = tri.simplices[idx]
        p1, p2, p3 = xy[verts[0]], xy[verts[1]], xy[verts[2]]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # 2倍三角形面积（带符号）
        twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(twoA) < 1e-30:
            twoA = 1e-30 * (1.0 if twoA >= 0 else -1.0)

        # 形函数导数
        B = np.zeros((2, 3))
        B[0, 0] = (y2 - y3) / twoA
        B[0, 1] = (y3 - y1) / twoA
        B[0, 2] = (y1 - y2) / twoA
        B[1, 0] = (x3 - x2) / twoA
        B[1, 1] = (x1 - x3) / twoA
        B[1, 2] = (x2 - x1) / twoA

        area = abs(twoA) / 2.0

        B_list.append(B)
        areas.append(area)

    return B_list, good_indices, np.array(areas)


# ---------------------------------------------------------------------------
# 三角网邻接关系
# ---------------------------------------------------------------------------

def build_adjacency(tri, good_mask):
    """
    构建三角网邻接关系

    两个三角形"邻接"当且仅当共享一条边

    参数
    ----
    tri : Delaunay
    good_mask : (N_tri,) bool array

    返回
    ----
    adjacency : dict {triangle_idx: set(neighbor_indices)}
        仅包含合格三角形的邻接关系
    """
    good_set = set(np.where(good_mask)[0])
    adjacency = {idx: set() for idx in good_set}

    # 构建边 → 三角形 映射
    edge_to_tris = {}

    for tri_idx in good_set:
        verts = tri.simplices[tri_idx]
        edges = [
            tuple(sorted((verts[0], verts[1]))),
            tuple(sorted((verts[1], verts[2]))),
            tuple(sorted((verts[2], verts[0]))),
        ]
        for edge in edges:
            if edge not in edge_to_tris:
                edge_to_tris[edge] = []
            edge_to_tris[edge].append(tri_idx)

    # 共享边 → 邻接
    for edge, tri_list in edge_to_tris.items():
        if len(tri_list) == 2:
            t1, t2 = tri_list
            adjacency[t1].add(t2)
            adjacency[t2].add(t1)

    return adjacency


# ---------------------------------------------------------------------------
# 站点覆盖检测
# ---------------------------------------------------------------------------

def detect_hanging_sites(tri, good_mask, n_sites):
    """
    检测"悬挂站点"——不属于任何合格三角形的站点

    返回
    ----
    hanging : 1D bool array
    """
    covered = np.zeros(n_sites, dtype=bool)
    for i in np.where(good_mask)[0]:
        for v in tri.simplices[i]:
            covered[v] = True
    return ~covered


# ---------------------------------------------------------------------------
# 站点密度控制
# ---------------------------------------------------------------------------

def thin_sites_by_spacing(lon, lat, se, sn, min_spacing_km):
    """
    按最小间距对站点抽稀，保留速度不确定度更小的站点（贪心算法）

    参数
    ----
    lon, lat : 1D array (度)
    se, sn   : 1D array (mm/yr)，东/北向速度不确定度
    min_spacing_km : float
        最小保留间距 (km)；两站 UTM 距离 < 此值时，剔除不确定度较大的那个

    返回
    ----
    keep_mask : 1D bool array
        True 表示保留，False 表示剔除
    """
    x, y, _ = _ll_to_utm(lon, lat)
    xy_km = np.column_stack([x / 1000.0, y / 1000.0])
    sigma = np.sqrt(se**2 + sn**2)

    # 按不确定度升序排列：精度好的优先保留
    order = np.argsort(sigma)
    keep = np.ones(len(lon), dtype=bool)
    tree = KDTree(xy_km)

    for i in order:
        if not keep[i]:
            continue
        # 找出以该站为圆心、min_spacing_km 为半径内的所有邻居
        neighbors = tree.query_ball_point(xy_km[i], min_spacing_km)
        for j in neighbors:
            if j != i:
                keep[j] = False

    return keep
