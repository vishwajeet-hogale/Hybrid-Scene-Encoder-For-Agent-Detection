from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def oriented_box_corners(x, y, w, l, yaw):
    """
    Returns 4 corners in (x,y) for a rectangle centered at (x,y)
    w: width (across y-ish), l: length (across x-ish) in ego plane
    yaw: rad, around z
    """
    c, s = np.cos(yaw), np.sin(yaw)
    # rectangle in local frame (length along x)
    dx = l / 2.0
    dy = w / 2.0
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
        [ dx,  dy],
    ])
    R = np.array([[c, -s],
                  [s,  c]])
    rot = corners @ R.T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot

def plot_bev_boxes(ax, boxes, scores=None, color=None, label_prefix=""):
    """
    boxes: [N,7] (x,y,z,w,l,h,yaw)
    """
    boxes = np.asarray(boxes)
    for i in range(boxes.shape[0]):
        x, y, z, w, l, h, yaw = boxes[i]
        poly = oriented_box_corners(x, y, w, l, yaw)
        ax.plot(poly[:, 0], poly[:, 1], linewidth=1.5)
        if scores is not None:
            ax.text(x, y, f"{label_prefix}{scores[i]:.2f}", fontsize=8)

def visualize_sample_bev(
    lidar_points,     # [N,4] tensor or np
    gt_boxes,         # [G,7]
    pred_boxes,       # [P,7]
    pred_scores=None,
    xlim=(-50, 50),
    ylim=(-50, 50),
):
    if hasattr(lidar_points, "cpu"):
        pts = lidar_points[:, :2].detach().cpu().numpy()
    else:
        pts = np.asarray(lidar_points)[:, :2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2)  # LiDAR XY

    if gt_boxes is not None and len(gt_boxes) > 0:
        if hasattr(gt_boxes, "cpu"):
            gt = gt_boxes.detach().cpu().numpy()
        else:
            gt = np.asarray(gt_boxes)
        plot_bev_boxes(ax, gt, scores=None, label_prefix="GT:")

    if pred_boxes is not None and len(pred_boxes) > 0:
        if hasattr(pred_boxes, "cpu"):
            pb = pred_boxes.detach().cpu().numpy()
        else:
            pb = np.asarray(pred_boxes)
        ps = None
        if pred_scores is not None:
            ps = pred_scores.detach().cpu().numpy() if hasattr(pred_scores, "cpu") else np.asarray(pred_scores)
        plot_bev_boxes(ax, pb, scores=ps, label_prefix="P:")

    ax.set_aspect("equal", "box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (m, ego)")
    ax.set_ylabel("y (m, ego)")
    ax.set_title("BEV: LiDAR points + GT boxes + Pred boxes")
    plt.show()
