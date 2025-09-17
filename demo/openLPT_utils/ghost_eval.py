# openlpt_utils/ghost_eval.py
import numpy as np
from collections import defaultdict

def find_ghosts(pts: np.ndarray,
                pts_reconstructed: np.ndarray,
                tol_mm: float = 2.0,
                use_ckdtree: bool = True):
    """
    Identify ghosts and build rec<->GT matching. Multiple rec -> same GT => repeated.

    Returns (additions):
      - rec_to_gt        : (N,)  每个重建点匹配到的 GT 索引（未匹配为 -1）： GT: ground true
      - gt_to_rec        : dict[int, np.ndarray] GT 索引 -> 命中的重建点索引数组
      - repeated_indices : (R,)  命中同一 GT 的“重复”重建点索引（可能多个）
      - repeated_mask    : (N,)  bool，重建点是否为“重复”
    """
    pts = np.asarray(pts, dtype=np.float64)
    rec = np.asarray(pts_reconstructed, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`pts` must be (M,3); got {pts.shape}")
    if rec.ndim != 2 or rec.shape[1] != 3:
        raise ValueError(f"`pts_reconstructed` must be (N,3); got {rec.shape}")

    N = len(rec)
    if N == 0:
        empty_i = np.array([], dtype=int)
        return {
            "ghost_indices": empty_i,
            "matched_indices": empty_i,
            "matched_gt_indices": empty_i,
            "dist_mm": np.array([], dtype=np.float64),
            "nn_indices": empty_i,
            "rec_to_gt": np.array([], dtype=int),
            "gt_to_rec": {},
            "repeated_indices": empty_i,
            "repeated_mask": np.array([], dtype=bool),
        }
    if len(pts) == 0:
        dist_mm = np.full((N,), np.inf)
        neg1 = np.full((N,), -1, dtype=int)
        return {
            "ghost_indices": np.arange(N, dtype=int),
            "matched_indices": np.array([], dtype=int),
            "matched_gt_indices": np.array([], dtype=int),
            "dist_mm": dist_mm,
            "nn_indices": neg1,
            "rec_to_gt": neg1,
            "gt_to_rec": {},
            "repeated_indices": np.array([], dtype=int),
            "repeated_mask": np.zeros((N,), dtype=bool),
        }

    # 最近邻（单位/尺度需与你的数据一致；tol_mm 即阈值）
    if use_ckdtree:
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(pts)
            dist_mm, nn_indices = tree.query(rec, k=1)
        except Exception:
            diff = rec[:, None, :] - pts[None, :, :]
            d2 = (diff * diff).sum(axis=2)
            nn_indices = d2.argmin(axis=1)
            dist_mm = np.sqrt(d2[np.arange(N), nn_indices])
    else:
        diff = rec[:, None, :] - pts[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        nn_indices = d2.argmin(axis=1)
        dist_mm = np.sqrt(d2[np.arange(N), nn_indices])

    is_match = dist_mm <= tol_mm
    matched_indices = np.where(is_match)[0]
    ghost_indices = np.where(~is_match)[0]
    matched_gt_indices = nn_indices[matched_indices]

    # ---- 新增：映射与重复判定 ----
    rec_to_gt = np.full((N,), -1, dtype=int)
    rec_to_gt[matched_indices] = matched_gt_indices

    gt_to_rec_dict: dict[int, list[int]] = defaultdict(list)
    for rec_i, gt_i in zip(matched_indices.tolist(), matched_gt_indices.tolist()):
        gt_to_rec_dict[int(gt_i)].append(int(rec_i))
    gt_to_rec = {k: np.array(v, dtype=int) for k, v in gt_to_rec_dict.items()}

    # 同一 GT 命中的多个重建点 => 这些重建点都视为 repeated
    repeated_list = [np.array(v, dtype=int) for v in gt_to_rec_dict.values() if len(v) >= 2]
    repeated_indices = (np.concatenate(repeated_list) if repeated_list
                        else np.array([], dtype=int))
    repeated_mask = np.zeros((N,), dtype=bool)
    if repeated_indices.size:
        repeated_mask[repeated_indices] = True

    return {
        "ghost_indices": ghost_indices,
        "matched_indices": matched_indices,
        "matched_gt_indices": matched_gt_indices,
        "dist_mm": dist_mm,
        "nn_indices": nn_indices,
        "rec_to_gt": rec_to_gt,
        "gt_to_rec": gt_to_rec,
        "repeated_indices": repeated_indices,
        "repeated_mask": repeated_mask,
    }
