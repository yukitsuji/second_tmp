import pickle

from second.core.preprocess import DataBasePreprocessor
from second.core.sample_ops import DataBaseSamplerV2
import second.core.preprocess as prep

def build_db_preprocess(db_prep_config):
    prep_type = db_prep_config.WhichOneof('database_preprocessing_step')

    if prep_type == 'filter_by_difficulty':
        cfg = db_prep_config.filter_by_difficulty
        return prep.DBFilterByDifficulty(list(cfg.removed_difficulties))
    elif prep_type == 'filter_by_min_num_points':
        cfg = db_prep_config.filter_by_min_num_points
        return prep.DBFilterByMinNumPoint(dict(cfg.min_num_point_pairs))
    else:
        raise ValueError("unknown database prep type")

def build(sampler_config):
    cfg = sampler_config
    groups = list(cfg.sample_groups)
    prepors = [
        build_db_preprocess(c)
        for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler
