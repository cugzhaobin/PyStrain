"""
gnss_strain.py — GNSS 速度场 → 应变率计算主程序

用法
----
    python gnss_strain.py --vel_file data.vel [--config my.yaml] [参数...]
    python gnss_strain.py --help
"""

import numpy as np
import sys
import os

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_utils import (read_velocity_file, sites_to_arrays,
                      write_strain_output, write_outliers, write_report)
from triangulation import (delaunay_triangulation,
                           build_adjacency, triangle_centroids_ll,
                           detect_hanging_sites, _ll_to_utm,
                           thin_sites_by_spacing)
from strain_rate import run_strain_rate_pipeline, compute_site_residuals
from uncertainty import monte_carlo_uncertainty
from outlier_detection import (knn_prescreening, iterative_outlier_removal)
from visualization import (plot_triangulation_and_velocities,
                           plot_scalar_field, plot_principal_strain)


# ---------------------------------------------------------------------------
# 三角剖分包装函数（供 outlier_detection 模块调用）
# ---------------------------------------------------------------------------

def make_triangulation_fn(polygon, min_angle, max_edge_pctl, max_edge_factor):
    """创建三角剖分函数（闭包绑定 poly + 参数）"""
    def fn(lon, lat, ve, vn, site_indices):
        tri, good_mask, xy, proj = delaunay_triangulation(
            lon[site_indices], lat[site_indices],
            polygon=polygon,
            min_angle_deg=min_angle,
            max_edge_pctl=max_edge_pctl,
            max_edge_factor=max_edge_factor,
        )
        return tri, xy, good_mask
    return fn


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_full_pipeline(vel_file=None, poly_file=None, output_dir='output',
                      vel_format='auto',
                      min_spacing_km=None,
                      max_edge_km=None,
                      smooth_weight=0.3, smooth_iter=2,
                      min_angle_deg=10.0, max_edge_pctl=95.0,
                      max_edge_factor=1.5,
                      mc_iterations=200,
                      k_neighbors=8, mad_factor=3.5, iqr_factor=1.5,
                      max_outlier_iter=5,
                      stage_callback=None):
    """
    完整的 GNSS 速度场 → 应变率计算流程

    参数
    ----
    vel_file : str
        速度场文件路径（必须提供）。
    poly_file : str or None
        多边形边界文件。None 则自动生成矩形边界。
    stage_callback : callable or None
        进度回调，签名 callback(stage: int, message: str)。
        供 Streamlit 等前端更新进度条，命令行传 None 即可。

    返回
    ----
    result : dict — 应变率结果
    unc    : dict — Monte Carlo 不确定度
    outlier_history : list — 被剔除站点记录
    """
    from io_utils import read_polygon

    def _cb(stage, msg):
        if stage_callback is not None:
            stage_callback(stage, msg)

    print("=" * 60)
    print("GNSS Strain Rate Calculation")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    if vel_file is not None and os.path.exists(vel_file):
        print(f"\n[1] Loading velocity file: {vel_file}  (format={vel_format})")
        _cb(1, f"Loading velocity file: {vel_file}")
        sites = read_velocity_file(vel_file, format=vel_format)
        lon, lat, ve, vn, se, sn, rho, names = sites_to_arrays(sites)
        n_original = len(ve)

        # ---- 1b. 站点抽稀（可选）----
        if min_spacing_km is not None:
            n_before = n_original
            keep = thin_sites_by_spacing(lon, lat, se, sn, min_spacing_km)
            lon, lat, ve, vn, se, sn, rho, names = (
                lon[keep], lat[keep], ve[keep], vn[keep],
                se[keep], sn[keep], rho[keep], names[keep])
            n_original = int(keep.sum())
            print(f"    Thinning (min_spacing={min_spacing_km} km): "
                  f"{n_before} → {n_original} sites")

        # 加载边界
        if poly_file is not None and os.path.exists(poly_file):
            print(f"    Loading polygon: {poly_file}")
            polygons = read_polygon(poly_file)
            polygon = polygons[0] if polygons else None
        else:
            print("    No polygon file, using convex hull of sites")
            polygon = np.array([
                [lon.min() - 0.02, lat.min() - 0.02],
                [lon.max() + 0.02, lat.min() - 0.02],
                [lon.max() + 0.02, lat.max() + 0.02],
                [lon.min() - 0.02, lat.max() + 0.02],
                [lon.min() - 0.02, lat.min() - 0.02],
            ])
    else:
        raise FileNotFoundError(
            f"Velocity file not found or not specified: {vel_file!r}\n"
            "Please provide a valid --vel_file path."
        )

    print(f"    Loaded {n_original} GPS sites")

    # ---- 2. KNN 预筛 ----
    print(f"\n[2] KNN pre-screening (k={k_neighbors}, mad={mad_factor})")
    _cb(2, "KNN pre-screening...")
    outlier_mask_knn, suspended = knn_prescreening(
        lon, lat, ve, vn, se, sn, names,
        k_neighbors=k_neighbors, mad_factor=mad_factor)
    print(f"    Outliers (KNN): {outlier_mask_knn.sum()}/{n_original}")
    print(f"    Suspended:     {suspended.sum()}/{n_original}")

    # ---- 3. 三角剖分 ----
    print(f"\n[3] Delaunay triangulation + quality control")
    print(f"    min_angle={min_angle_deg}°, max_edge=p{max_edge_pctl}×{max_edge_factor}"
          + (f", max_edge_km={max_edge_km}" if max_edge_km is not None else ""))
    _cb(3, "Delaunay triangulation...")

    # 初筛后保留的站点
    keep_mask_initial = ~outlier_mask_knn & ~suspended
    site_indices_initial = np.where(keep_mask_initial)[0]

    tri, good_mask, xy, proj_params = delaunay_triangulation(
        lon[site_indices_initial], lat[site_indices_initial],
        polygon=polygon,
        min_angle_deg=min_angle_deg,
        max_edge_pctl=max_edge_pctl,
        max_edge_factor=max_edge_factor,
        max_edge_km=max_edge_km,
    )

    n_tri_total = tri.simplices.shape[0]
    n_good = good_mask.sum()
    print(f"    Total triangles: {n_tri_total}")
    print(f"    Good triangles:  {n_good} ({100*n_good/max(n_tri_total,1):.1f}%)")

    if n_good < 3:
        raise RuntimeError(f"Only {n_good} valid triangles. Relax constraints!")

    # ---- 4. 异常值迭代剔除 ----
    print(f"\n[4] Iterative outlier removal (max_iter={max_outlier_iter})")
    _cb(4, "Iterative outlier removal...")

    triangulation_fn = make_triangulation_fn(
        polygon, min_angle_deg, max_edge_pctl, max_edge_factor)

    # 合并 KNN + suspended 的初始 mask
    initial_bad = outlier_mask_knn | suspended

    ve_clean = ve.copy()
    vn_clean = vn.copy()

    # 迭代异常值检测
    final_keep_mask, outlier_history = iterative_outlier_removal(
        lon, lat, ve_clean, vn_clean, se, sn, rho, names,
        triangulation_fn,
        k_neighbors=k_neighbors, mad_factor=mad_factor,
        iqr_factor=iqr_factor, max_iterations=max_outlier_iter,
    )

    # 初始异常值也加进来
    initial_outliers = np.where(initial_bad)[0]
    for s in initial_outliers:
        if final_keep_mask[s]:
            outlier_history.append({
                'name': names[s] if names is not None else f'site_{s}',
                'lon': lon[s], 'lat': lat[s],
                'residual': 0.0,
                'reason': 'KNN_initial',
                'iteration': -1,
            })
            final_keep_mask[s] = False

    n_removed = n_original - final_keep_mask.sum()
    n_used = final_keep_mask.sum()
    print(f"    Removed: {n_removed}")
    print(f"    Used:    {n_used}")

    # 用最终保留站点重做三角剖分
    site_indices_final = np.where(final_keep_mask)[0]
    tri, good_mask, xy, proj_params = delaunay_triangulation(
        lon[site_indices_final], lat[site_indices_final],
        polygon=polygon,
        min_angle_deg=min_angle_deg,
        max_edge_pctl=max_edge_pctl,
        max_edge_factor=max_edge_factor,
        max_edge_km=max_edge_km,
    )
    n_good = good_mask.sum()
    print(f"    Final good triangles: {n_good}")

    # ---- 5. 应变率计算 ----
    print(f"\n[5] Computing strain rate")
    _cb(5, "Computing strain rate...")
    result, adjacency = run_strain_rate_pipeline(
        tri, xy, good_mask,
        ve[site_indices_final], vn[site_indices_final],
        lon[site_indices_final], lat[site_indices_final],
        smooth_weight=smooth_weight, smooth_iter=smooth_iter,
    )
    print(f"    Strain rate computed for {len(result['tri_ids'])} triangles")
    print(f"    Dilatation range: {result['dilatation'].min():.1f} to "
          f"{result['dilatation'].max():.1f} nstrain/yr")
    print(f"    Max shear range:  {result['max_shear'].min():.1f} to "
          f"{result['max_shear'].max():.1f} nstrain/yr")

    # ---- 6. 不确定性 ----
    print(f"\n[6] Monte Carlo uncertainty (iterations={mc_iterations})")
    _cb(6, f"Monte Carlo uncertainty ({mc_iterations} iterations)...")
    unc = monte_carlo_uncertainty(
        tri, xy, good_mask,
        ve[site_indices_final], vn[site_indices_final],
        se[site_indices_final], sn[site_indices_final],
        rho[site_indices_final],
        n_iterations=mc_iterations,
    )

    # 合并不确定性到 result
    result.update({
        'e_ee_std': unc['e_ee_std'], 'e_en_std': unc['e_en_std'],
        'e_nn_std': unc['e_nn_std'],
        'e1_std': unc['e1_std'], 'e2_std': unc['e2_std'],
        'dilatation_std': unc['dilatation_std'],
        'max_shear_std': unc['max_shear_std'],
    })

    print(f"    Mean e_ee uncertainty: {unc['e_ee_std'].mean():.1f} nstrain/yr")
    print(f"    Mean e_nn uncertainty: {unc['e_nn_std'].mean():.1f} nstrain/yr")

    # ---- 7. 输出 ----
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print(f"\n[7] Writing output to {output_dir}/")
    _cb(7, f"Writing output to {output_dir}/")

    write_strain_output(os.path.join(output_dir, 'strain_triangles.dat'), result)
    write_outliers(os.path.join(output_dir, 'outliers.txt'), outlier_history)

    stats = {
        'n_input': n_original,
        'n_removed': n_removed,
        'n_used': n_used,
        'n_tri': n_good,
        'smooth_w': smooth_weight,
        'mc_iterations': mc_iterations,
        'min_angle': min_angle_deg,
        'max_edge_pctl': max_edge_pctl,
    }
    write_report(os.path.join(output_dir, 'report.txt'), stats)

    # ---- 8. 绘图 ----
    print(f"\n[8] Generating figures...")
    _cb(8, "Generating figures...")

    all_outlier_mask = ~final_keep_mask

    plot_triangulation_and_velocities(
        tri, xy, lon[site_indices_final], lat[site_indices_final],
        ve[site_indices_final], vn[site_indices_final],
        good_mask=good_mask, polygon=polygon,
        title='Triangulation & GPS Velocities',
        save_path=os.path.join(output_dir, 'figures', 'triangulation.png'),
    )

    plot_scalar_field(
        lon[site_indices_final], lat[site_indices_final],
        tri, good_mask, result['dilatation'],
        title='Dilatation Rate', cmap='RdBu_r',
        cbar_label='Dilatation',
        save_path=os.path.join(output_dir, 'figures', 'dilatation.png'),
    )

    plot_scalar_field(
        lon[site_indices_final], lat[site_indices_final],
        tri, good_mask, result['max_shear'],
        title='Maximum Shear Strain Rate', cmap='YlOrRd',
        cbar_label='Max Shear',
        save_path=os.path.join(output_dir, 'figures', 'max_shear.png'),
    )

    plot_principal_strain(
        tri, lon[site_indices_final], lat[site_indices_final],
        good_mask,
        result['centroids_lon'], result['centroids_lat'],
        result['e1'], result['e2'], result['azimuth'],
        title='Principal Strain Rate',
        save_path=os.path.join(output_dir, 'figures', 'principal_strain.png'),
    )

    print(f"    Done. Figures saved to {output_dir}/figures/")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    # 将关键数据存入 result，供前端调用
    result['_lon'] = lon[site_indices_final]
    result['_lat'] = lat[site_indices_final]
    result['_ve'] = ve[site_indices_final]
    result['_vn'] = vn[site_indices_final]
    result['_se'] = se[site_indices_final]
    result['_sn'] = sn[site_indices_final]
    result['_names'] = names[site_indices_final]
    result['_tri'] = tri
    result['_xy'] = xy
    result['_good_mask'] = good_mask
    result['_polygon'] = polygon
    result['_outlier_lon'] = lon[~final_keep_mask]
    result['_outlier_lat'] = lat[~final_keep_mask]

    return result, unc, outlier_history


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from config import load_config, print_effective_config, cfg_to_kwargs

    parser = argparse.ArgumentParser(
        description='GNSS 速度场 → 应变率计算',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',          default=None,  metavar='FILE',
                        help='YAML 配置文件路径')
    # 数据
    parser.add_argument('--vel_file',        default=None,  metavar='FILE',
                        help='速度场文件路径')
    parser.add_argument('--poly_file',       default=None,  metavar='FILE',
                        help='边界多边形文件路径')
    parser.add_argument('--output_dir',      default=None,  metavar='DIR',
                        help='输出目录')
    parser.add_argument('--format',          default=None,  dest='vel_format',
                        choices=['auto', 'gmt', 'globk'],
                        help='输入文件格式 (auto|gmt|globk)')
    # 光滑
    parser.add_argument('--smooth_weight',   default=None,  type=float,
                        help='光滑权重 [0,1]')
    parser.add_argument('--smooth_iter',     default=None,  type=int,
                        help='光滑迭代次数')
    # 三角网
    parser.add_argument('--min_angle_deg',   default=None,  type=float,
                        help='最小三角形角度 (°)')
    parser.add_argument('--max_edge_pctl',   default=None,  type=float,
                        help='最大边长百分位数')
    parser.add_argument('--max_edge_factor', default=None,  type=float,
                        help='最大边长倍数因子')
    parser.add_argument('--min_spacing_km',  default=None,  type=float,
                        help='站点最小间距 (km)，启用站点抽稀')
    parser.add_argument('--max_edge_km',     default=None,  type=float,
                        help='三角形边长绝对上限 (km)')
    # 不确定度
    parser.add_argument('--mc_iterations',   default=None,  type=int,
                        help='Monte Carlo 迭代次数')
    # 异常值
    parser.add_argument('--k_neighbors',     default=None,  type=int,
                        help='KNN 邻居数')
    parser.add_argument('--mad_factor',      default=None,  type=float,
                        help='MAD 阈值倍数')
    parser.add_argument('--iqr_factor',      default=None,  type=float,
                        help='IQR 阈值倍数')
    parser.add_argument('--max_outlier_iter',default=None,  type=int,
                        help='异常值迭代最大次数')
    args = parser.parse_args()

    # 只将显式传入的参数（非 None）作为覆盖项
    overrides = {k: v for k, v in vars(args).items()
                 if k != 'config' and v is not None}

    cfg = load_config(args.config, overrides)
    print_effective_config(cfg)

    run_full_pipeline(**cfg_to_kwargs(cfg))
    print("\nAll done.")
