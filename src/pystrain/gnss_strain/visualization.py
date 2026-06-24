"""
visualization.py — 应变率结果可视化

绘图：
1. 三角网 + 速度矢量图
2. 面膨胀率分布图
3. 最大剪应变率分布图
4. 主应变率十字图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_triangulation_and_velocities(tri, xy, lon, lat, ve, vn,
                                      good_mask=None, polygon=None,
                                      outlier_mask=None,
                                      title='Triangulation & GPS Velocities',
                                      save_path=None):
    """
    三角网 + 速度矢量图

    参数
    ----
    tri : Delaunay
    xy : (N,2) UTM 坐标
    lon, lat : (N,) 经纬度
    ve, vn : (N,) 速度 (mm/yr)
    good_mask : (N_tri,) bool — 合格三角形
    polygon : (M,2) array — 边界
    outlier_mask : (N,) bool — 异常站点
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制合格三角形
    if good_mask is not None:
        good_indices = np.where(good_mask)[0]
        # 用经纬度绘制三角形（近似）
        good_tri_ll = lon[tri.simplices[good_indices]]
        tri_verts_ll = lat[tri.simplices[good_indices]]
        for i in range(len(good_indices)):
            xs = good_tri_ll[i]
            ys = tri_verts_ll[i]
            ax.fill(xs, ys, facecolor='none', edgecolor='gray',
                    linewidth=0.5, alpha=0.6)

    # 不合格三角形（红色虚线）
    if good_mask is not None:
        bad_indices = np.where(~good_mask)[0]
        if len(bad_indices) > 0:
            bad_tri_ll = lon[tri.simplices[bad_indices]]
            bad_tri_lat = lat[tri.simplices[bad_indices]]
            for i in range(len(bad_indices)):
                xs = bad_tri_ll[i]
                ys = bad_tri_lat[i]
                ax.fill(xs, ys, facecolor='none', edgecolor='red',
                        linewidth=0.3, alpha=0.3, linestyle=':')

    # 多边形边界
    if polygon is not None:
        ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=1.5, label='Boundary')

    # GNSS 速度矢量
    normal_mask = np.ones(len(lon), dtype=bool)
    if outlier_mask is not None:
        normal_mask = ~outlier_mask

    scale = np.percentile(np.sqrt(ve[normal_mask]**2 + vn[normal_mask]**2), 95) * 20
    if scale < 1e-6:
        scale = 1.0

    ax.quiver(lon[normal_mask], lat[normal_mask],
              ve[normal_mask], vn[normal_mask],
              scale=scale, scale_units='inches', width=0.003,
              color='blue', alpha=0.7, label='GPS velocity')

    # 异常值（红色）
    if outlier_mask is not None and outlier_mask.any():
        ax.quiver(lon[outlier_mask], lat[outlier_mask],
                  ve[outlier_mask], vn[outlier_mask],
                  scale=scale, scale_units='inches', width=0.003,
                  color='red', alpha=0.9, label='Outlier')

    # 参考箭头
    ax.quiver(lon.max() + 0.05, lat.min() + 0.02,
              1, 0, scale=scale, scale_units='inches', width=0.003,
              color='black')
    ax.text(lon.max() + 0.05, lat.min() - 0.01, '1 mm/yr',
            fontsize=8, ha='center')

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat))))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def plot_scalar_field(lon, lat, tri, good_mask, scalar, title='',
                      cmap='RdBu_r', cbar_label='', vmin=None, vmax=None,
                      units_str='nstrain/yr', save_path=None):
    """
    三角形标量场颜色图

    参数
    ----
    lon, lat : (N,) 站点经纬度
    tri : Delaunay
    good_mask : (N_tri,) bool
    scalar : (N_good,) — 每个三角形的标量值
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    good_indices = np.where(good_mask)[0]
    n_good = len(good_indices)

    if vmin is None:
        vmin = np.percentile(scalar, 2)
    if vmax is None:
        vmax = np.percentile(scalar, 98)

    # 构建三角形顶点集合
    verts_list = []
    colors_list = []
    for k, tri_idx in enumerate(good_indices):
        v = tri.simplices[tri_idx]
        verts = np.column_stack([lon[v], lat[v]])
        verts_list.append(verts)
        colors_list.append(scalar[k])

    norm = Normalize(vmin=vmin, vmax=vmax)
    coll = PolyCollection(verts_list, array=np.array(colors_list),
                          cmap=cmap, norm=norm, edgecolor='none', alpha=0.9)
    ax.add_collection(coll)

    cbar = plt.colorbar(coll, ax=ax, shrink=0.8)
    cbar.set_label(f'{cbar_label} ({units_str})')

    ax.set_xlim(lon.min() - 0.05, lon.max() + 0.05)
    ax.set_ylim(lat.min() - 0.05, lat.max() + 0.05)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat))))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def plot_principal_strain(tri, lon, lat, good_mask,
                          centroids_lon, centroids_lat,
                          e1, e2, azimuth,
                          scale_factor=0.5, title='Principal Strain Rate',
                          save_path=None):
    """
    主应变率十字图

    交叉十字：水平线沿 ε1 方向（拉张），垂直线沿 ε2 方向（压缩）

    长度 = |ε| * scale_factor

    参数
    ----
    e1, e2 : 主应变率（ε1 拉张=正, ε2 压缩=负）
    azimuth : ε1 方向（从东逆时针，°）
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    good_indices = np.where(good_mask)[0]

    # 先画三角形底图
    for tri_idx in good_indices:
        v = tri.simplices[tri_idx]
        xs = lon[v]
        ys = lat[v]
        ax.fill(xs, ys, facecolor='none', edgecolor='lightgray',
                linewidth=0.3, alpha=0.5)

    # 主应变十字
    az_rad = np.radians(azimuth)

    if scale_factor is None:
        # 自动缩放
        scale_factor = 0.5 / (np.percentile(np.maximum(np.abs(e1), np.abs(e2)), 95) + 1e-30)

    for k in range(len(centroids_lon)):
        cx = centroids_lon[k]
        cy = centroids_lat[k]

        # ε1 方向（拉张）：线段在 azimuth 方向
        dx1 = np.cos(az_rad[k]) * e1[k] * scale_factor
        dy1 = np.sin(az_rad[k]) * e1[k] * scale_factor

        if e1[k] >= 0:
            # 拉张 → 黑色实线（两段，向外）
            ax.plot([cx - dx1, cx + dx1], [cy - dy1, cy + dy1],
                    'k-', linewidth=1.0, alpha=0.8)
        else:
            # 压缩的 ε1 方向 → 红色虚线
            ax.plot([cx - dx1, cx + dx1], [cy - dy1, cy + dy1],
                    'r--', linewidth=1.0, alpha=0.8)

        # ε2 方向（垂直于 ε1）
        dx2 = -np.sin(az_rad[k]) * e2[k] * scale_factor
        dy2 = np.cos(az_rad[k]) * e2[k] * scale_factor

        if e2[k] <= 0:
            # 压缩 → 红色实线
            ax.plot([cx - dx2, cx + dx2], [cy - dy2, cy + dy2],
                    'r-', linewidth=1.0, alpha=0.8)
        else:
            # 拉张的 ε2 方向 → 黑色虚线
            ax.plot([cx - dx2, cx + dx2], [cy - dy2, cy + dy2],
                    'k--', linewidth=1.0, alpha=0.8)

    legend_items = [
        plt.Line2D([0], [0], color='k', linewidth=1.5, label='Extension'),
        plt.Line2D([0], [0], color='r', linewidth=1.5, label='Compression'),
    ]
    ax.legend(handles=legend_items, loc='upper right')

    ax.set_xlim(lon.min() - 0.05, lon.max() + 0.05)
    ax.set_ylim(lat.min() - 0.05, lat.max() + 0.05)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(title)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lat))))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax
