"""
io_utils.py — 读写 GNSS 速度场文件和边界多边形文件

速度场格式 (.vel):
    lon lat Ve Vn Se Sn Rho SiteName
    - Ve, Vn: 东向/北向速度 (mm/yr)
    - Se, Sn: 东向/北向速度不确定度 (mm/yr)
    - Rho: 相关系数

多边形边界格式 (.poly):
    每行 lon lat，闭合多边形（最后一点应与第一点相同或自动闭合）
    空行分隔不同的多边形环（外边界 + 空洞）
"""

import numpy as np
from collections import namedtuple

SiteData = namedtuple('SiteData', ['lon', 'lat', 've', 'vn', 'se', 'sn', 'rho', 'name'])


def read_velocity_file(filepath, format='auto'):
    """
    读取 GNSS 速度场文件

    支持格式
    --------
    gmt   (8列):  lon lat Ve Vn Se Sn Rho SiteName
    globk (13列): lon lat ve vn Ve_adj Vn_adj Se Sn Rho Vu Vu_adj Su site
    auto  : 按列数自动判断 (8列→gmt, 13列→globk, 其他→fallback)

    参数
    ----
    filepath : str
        速度场文件路径
    format : str
        'auto' | 'gmt' | 'globk'

    返回
    ----
    sites : list of SiteData
        站点数据列表
    """
    format = format.lower() if format else 'auto'
    sites = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ncols = len(parts)
            if ncols < 7:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])

                # 判断实际使用的格式
                use_globk = (format == 'globk') or (format == 'auto' and ncols == 13)
                use_gmt   = (format == 'gmt')   or (format == 'auto' and ncols == 8)

                if use_globk:
                    # GLOBK 13列: lon lat ve vn Ve_adj Vn_adj Se Sn Rho Vu Vu_adj Su site
                    if ncols < 13:
                        continue  # 列数不足，跳过
                    ve   = float(parts[2])   # 原始东向速度
                    vn   = float(parts[3])   # 原始北向速度
                    se   = float(parts[6])   # 东向不确定度
                    sn   = float(parts[7])   # 北向不确定度
                    rho  = float(parts[8])   # 相关系数
                    name = parts[12]          # 站点名（第13列）
                elif use_gmt or ncols == 7:
                    if ncols == 7:
                        # 7列: lon lat Ve Vn Se Sn Site
                        ve = float(parts[2])
                        vn = float(parts[3])
                        se = float(parts[4])
                        sn = float(parts[5])
                        rho = 0.0
                        name = parts[6]
                    else:
                        # GMT 8列: lon lat Ve Vn Se Sn Rho Site
                        ve = float(parts[2])
                        vn = float(parts[3])
                        se = float(parts[4])
                        sn = float(parts[5])
                        rho = float(parts[6])
                        name = parts[7]
                elif ncols >= 12:
                    # GMT vector format (12+ cols):
                    # lon lat Ve Vn Ve_abs Vn_abs Se Sn Rho X Y Z Site
                    ve = float(parts[2])
                    vn = float(parts[3])
                    se = float(parts[6])
                    sn = float(parts[7])
                    rho = float(parts[8])
                    name = parts[-1]
                else:
                    # fallback
                    ve = float(parts[2])
                    vn = float(parts[3])
                    se = float(parts[4]) if ncols > 4 else 1.0
                    sn = float(parts[5]) if ncols > 5 else 1.0
                    rho = float(parts[6]) if ncols > 6 else 0.0
                    name = parts[-1] if len(parts) > 7 else ''
                sites.append(SiteData(lon, lat, ve, vn, se, sn, rho, name))
            except ValueError:
                continue
    return sites


def sites_to_arrays(sites):
    """
    将站点列表转换为 numpy 数组

    参数
    ----
    sites : list of SiteData

    返回
    ----
    lon, lat, ve, vn, se, sn, rho, names : 1D numpy arrays
    """
    lon = np.array([s.lon for s in sites])
    lat = np.array([s.lat for s in sites])
    ve = np.array([s.ve for s in sites])
    vn = np.array([s.vn for s in sites])
    se = np.array([s.se for s in sites])
    sn = np.array([s.sn for s in sites])
    rho = np.array([s.rho for s in sites])
    names = np.array([s.name for s in sites])
    return lon, lat, ve, vn, se, sn, rho, names


def filter_sites_by_indices(sites, indices):
    """按索引筛选站点"""
    return [sites[i] for i in indices]


def read_polygon(filepath):
    """
    读取多边形边界文件

    格式：每行 lon lat，空行分隔不同的环

    参数
    ----
    filepath : str
        多边形文件路径

    返回
    ----
    polygons : list of numpy arrays
        每个元素是 (N, 2) 的顶点数组
    """
    polygons = []
    current_ring = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_ring:
                    polygons.append(np.array(current_ring))
                    current_ring = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    current_ring.append([lon, lat])
                except ValueError:
                    continue
        if current_ring:
            polygons.append(np.array(current_ring))

    # 自动闭合未闭合的多边形
    for i, poly in enumerate(polygons):
        if len(poly) > 2:
            if poly[0, 0] != poly[-1, 0] or poly[0, 1] != poly[-1, 1]:
                polygons[i] = np.vstack([poly, poly[0:1, :]])

    return polygons


def write_strain_output(filepath, tri_data):
    """
    输出三角形应变率结果到文本文件

    参数
    ----
    filepath : str
    tri_data : dict with keys:
        centroids_lon, centroids_lat, e_ee, e_en, e_nn,
        e1, e2, azimuth, dilatation, max_shear, second_invariant,
        [e_ee_std, e_en_std, e_nn_std, e1_std, e2_std, dilatation_std, max_shear_std]
        tri_ids
    """
    header = (
        "# tri_id  lon  lat  e_ee  e_en  e_nn  "
        "e1  e2  azimuth  dilatation  max_shear  second_invariant"
    )
    has_unc = 'e_ee_std' in tri_data

    if has_unc:
        header += (
            "  e_ee_std  e_en_std  e_nn_std  "
            "e1_std  e2_std  dilatation_std  max_shear_std"
        )

    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for i in range(len(tri_data['tri_ids'])):
            line = (
                f"{tri_data['tri_ids'][i]:5d}  "
                f"{tri_data['centroids_lon'][i]:10.5f}  {tri_data['centroids_lat'][i]:9.5f}  "
                f"{tri_data['e_ee'][i]:12.3f}  {tri_data['e_en'][i]:12.3f}  {tri_data['e_nn'][i]:12.3f}  "
                f"{tri_data['e1'][i]:12.3f}  {tri_data['e2'][i]:12.3f}  {tri_data['azimuth'][i]:8.2f}  "
                f"{tri_data['dilatation'][i]:12.3f}  {tri_data['max_shear'][i]:12.3f}  "
                f"{tri_data['second_invariant'][i]:12.3f}"
            )
            if has_unc:
                line += (
                    f"  {tri_data['e_ee_std'][i]:10.3f}  {tri_data['e_en_std'][i]:10.3f}  "
                    f"{tri_data['e_nn_std'][i]:10.3f}  {tri_data['e1_std'][i]:10.3f}  "
                    f"{tri_data['e2_std'][i]:10.3f}  {tri_data['dilatation_std'][i]:10.3f}  "
                    f"{tri_data['max_shear_std'][i]:10.3f}"
                )
            f.write(line + '\n')


def write_outliers(filepath, outliers):
    """
    输出异常值列表

    参数
    ----
    filepath : str
    outliers : list of dict
        每个元素包含 {name, lon, lat, residual, reason}
    """
    with open(filepath, 'w') as f:
        f.write("# Outlier report\n")
        f.write("# name  lon  lat  residual  reason\n")
        for o in outliers:
            f.write(f"{o['name']:12s}  {o['lon']:10.5f}  {o['lat']:9.5f}  "
                    f"{o['residual']:10.3f}  {o['reason']}\n")


def write_report(filepath, stats):
    """
    输出统计摘要

    参数
    ----
    filepath : str
    stats : dict
    """
    with open(filepath, 'w') as f:
        f.write("# GNSS Strain Rate Calculation Report\n")
        f.write(f"Number of input sites:      {stats.get('n_input', 0)}\n")
        f.write(f"Number of sites removed:    {stats.get('n_removed', 0)}\n")
        f.write(f"Number of sites used:       {stats.get('n_used', 0)}\n")
        f.write(f"Number of triangles:        {stats.get('n_tri', 0)}\n")
        f.write(f"Smoothing weight:           {stats.get('smooth_w', 0)}\n")
        f.write(f"Monte Carlo iterations:     {stats.get('mc_iterations', 0)}\n")
        f.write(f"Alpha-shape radius:         {stats.get('alpha', 'N/A')}\n")
        f.write(f"Min triangle angle (deg):   {stats.get('min_angle', 0)}\n")
        f.write(f"Max edge length percentile: {stats.get('max_edge_pctl', 0)}\n")
