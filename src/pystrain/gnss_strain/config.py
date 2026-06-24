"""
config.py — GNSS 应变率计算配置管理

功能：
1. 内置默认参数
2. 从 YAML 文件加载配置（深度合并覆盖默认值）
3. 应用命令行覆盖参数
4. 参数校验
5. 打印生效配置
"""

import copy

# ---------------------------------------------------------------------------
# 内置默认值
# ---------------------------------------------------------------------------

DEFAULTS = {
    'data': {
        'vel_file':  './camp_eura.vel',
        'poly_file': None,
        'output_dir': './output',
        'format':    'auto',   # 'auto' | 'gmt' | 'globk'
    },
    'outlier_detection': {
        'k_neighbors':    8,
        'mad_factor':     3.5,
        'iqr_factor':     1.5,
        'max_iterations': 5,
    },
    'triangulation': {
        'min_angle_deg':   10.0,
        'max_edge_pctl':   95.0,
        'max_edge_factor': 1.5,
        'min_spacing_km':  None,   # None = 不启用站点抽稀
        'max_edge_km':     None,   # None = 不启用绝对边长上限
    },
    'smoothing': {
        'weight':     0.3,
        'iterations': 3,
    },
    'uncertainty': {
        'mc_iterations': 200,
    },
    'visualization': {
        'dpi':          150,
        'save_figures': True,
        'show_figures': False,
    },
}

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path=None, overrides=None):
    """
    三层合并：内置默认值 → YAML 文件 → 命令行覆盖

    参数
    ----
    config_path : str or None
        YAML 配置文件路径；None 表示只用默认值
    overrides : dict or None
        命令行覆盖参数（扁平化键名，如 'smooth_weight', 'min_spacing_km' 等）

    返回
    ----
    cfg : dict  嵌套配置字典
    """
    cfg = copy.deepcopy(DEFAULTS)

    # 1. 合并 YAML
    if config_path is not None:
        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装 PyYAML：pip install pyyaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg, config_path)

    # 2. 应用命令行覆盖
    if overrides:
        _apply_overrides(cfg, overrides)

    # 3. 校验
    validate_config(cfg)

    return cfg


def _deep_merge(base, user, source=''):
    """将 user 字典深度合并到 base（仅更新已知键，未知键打警告）"""
    for section, values in user.items():
        if section not in base:
            print(f"WARNING: 配置文件 '{source}' 中未知节 '{section}'，已忽略")
            continue
        if not isinstance(values, dict):
            print(f"WARNING: 配置节 '{section}' 应为字典，已忽略")
            continue
        for key, val in values.items():
            if key not in base[section]:
                print(f"WARNING: 配置文件 '{source}' 中未知参数 '{section}.{key}'，已忽略")
            else:
                base[section][key] = val


# 命令行扁平键名 → (section, key) 映射
_CLI_KEY_MAP = {
    'vel_file':        ('data',             'vel_file'),
    'poly_file':       ('data',             'poly_file'),
    'output_dir':      ('data',             'output_dir'),
    'vel_format':      ('data',             'format'),
    'smooth_weight':   ('smoothing',        'weight'),
    'smooth_iter':     ('smoothing',        'iterations'),
    'min_angle_deg':   ('triangulation',    'min_angle_deg'),
    'max_edge_pctl':   ('triangulation',    'max_edge_pctl'),
    'max_edge_factor': ('triangulation',    'max_edge_factor'),
    'min_spacing_km':  ('triangulation',    'min_spacing_km'),
    'max_edge_km':     ('triangulation',    'max_edge_km'),
    'mc_iterations':   ('uncertainty',      'mc_iterations'),
    'k_neighbors':     ('outlier_detection','k_neighbors'),
    'mad_factor':      ('outlier_detection','mad_factor'),
    'iqr_factor':      ('outlier_detection','iqr_factor'),
    'max_outlier_iter':('outlier_detection','max_iterations'),
}


def _apply_overrides(cfg, overrides):
    for key, val in overrides.items():
        if key in _CLI_KEY_MAP:
            section, param = _CLI_KEY_MAP[key]
            cfg[section][param] = val
        else:
            print(f"WARNING: 未知命令行参数 '{key}'，已忽略")


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def validate_config(cfg):
    """校验配置参数范围，违规时抛 ValueError"""
    errors = []

    def _check(cond, msg):
        if not cond:
            errors.append(msg)

    t = cfg['triangulation']
    s = cfg['smoothing']
    o = cfg['outlier_detection']
    u = cfg['uncertainty']
    v = cfg['visualization']

    # triangulation
    _check(0 < t['min_angle_deg'] < 60,
           f"triangulation.min_angle_deg={t['min_angle_deg']} 必须在 (0, 60) 范围内")
    _check(0 < t['max_edge_pctl'] < 100,
           f"triangulation.max_edge_pctl={t['max_edge_pctl']} 必须在 (0, 100) 范围内")
    _check(t['max_edge_factor'] > 1.0,
           f"triangulation.max_edge_factor={t['max_edge_factor']} 必须 > 1.0")
    _check(t['min_spacing_km'] is None or t['min_spacing_km'] > 0,
           f"triangulation.min_spacing_km={t['min_spacing_km']} 必须 > 0 或 null")
    _check(t['max_edge_km'] is None or t['max_edge_km'] > 0,
           f"triangulation.max_edge_km={t['max_edge_km']} 必须 > 0 或 null")

    # smoothing
    _check(0.0 <= s['weight'] <= 1.0,
           f"smoothing.weight={s['weight']} 必须在 [0, 1] 范围内")
    _check(isinstance(s['iterations'], int) and s['iterations'] >= 0,
           f"smoothing.iterations={s['iterations']} 必须为非负整数")

    # outlier_detection
    _check(isinstance(o['k_neighbors'], int) and o['k_neighbors'] >= 3,
           f"outlier_detection.k_neighbors={o['k_neighbors']} 必须为整数且 ≥ 3")
    _check(o['mad_factor'] > 0,
           f"outlier_detection.mad_factor={o['mad_factor']} 必须 > 0")
    _check(o['iqr_factor'] > 0,
           f"outlier_detection.iqr_factor={o['iqr_factor']} 必须 > 0")
    _check(isinstance(o['max_iterations'], int) and o['max_iterations'] >= 1,
           f"outlier_detection.max_iterations={o['max_iterations']} 必须为整数且 ≥ 1")

    # uncertainty
    _check(isinstance(u['mc_iterations'], int) and u['mc_iterations'] >= 10,
           f"uncertainty.mc_iterations={u['mc_iterations']} 必须为整数且 ≥ 10")

    # visualization
    _check(isinstance(v['dpi'], int) and v['dpi'] >= 50,
           f"visualization.dpi={v['dpi']} 必须为整数且 ≥ 50")

    if errors:
        raise ValueError("配置参数错误：\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# 打印生效配置
# ---------------------------------------------------------------------------

def print_effective_config(cfg):
    """以 YAML 格式打印当前生效配置"""
    try:
        import yaml
        text = yaml.dump(cfg, allow_unicode=True, default_flow_style=False,
                         sort_keys=False)
    except ImportError:
        # 降级为简单打印
        text = str(cfg)
    sep = '=' * 50
    print(f"\n{sep}")
    print("Effective Configuration")
    print(sep)
    print(text.rstrip())
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 配置展平 → run_full_pipeline 关键字参数
# ---------------------------------------------------------------------------

def cfg_to_kwargs(cfg):
    """将嵌套配置字典展平为 run_full_pipeline() 的参数字典"""
    return dict(
        vel_file         = cfg['data']['vel_file'],
        poly_file        = cfg['data']['poly_file'],
        output_dir       = cfg['data']['output_dir'],
        vel_format       = cfg['data']['format'],
        smooth_weight    = cfg['smoothing']['weight'],
        smooth_iter      = cfg['smoothing']['iterations'],
        min_angle_deg    = cfg['triangulation']['min_angle_deg'],
        max_edge_pctl    = cfg['triangulation']['max_edge_pctl'],
        max_edge_factor  = cfg['triangulation']['max_edge_factor'],
        min_spacing_km   = cfg['triangulation']['min_spacing_km'],
        max_edge_km      = cfg['triangulation']['max_edge_km'],
        mc_iterations    = cfg['uncertainty']['mc_iterations'],
        k_neighbors      = cfg['outlier_detection']['k_neighbors'],
        mad_factor       = cfg['outlier_detection']['mad_factor'],
        iqr_factor       = cfg['outlier_detection']['iqr_factor'],
        max_outlier_iter = cfg['outlier_detection']['max_iterations'],
    )
