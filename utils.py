import numpy as np

def calc_intersection_volume(bbox_a, bbox_b):
    """
    计算两个扁平 BBox [x1, y1, z1, x2, y2, z2] 的相交体积
    """
    # 提取坐标
    ax1, ay1, az1, ax2, ay2, az2 = bbox_a
    bx1, by1, bz1, bx2, by2, bz2 = bbox_b
    
    # 计算相交区域
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    iz1 = max(az1, bz1)
    
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iz2 = min(az2, bz2)
    
    # 如果没有相交
    if ix1 >= ix2 or iy1 >= iy2 or iz1 >= iz2:
        return 0
    
    return (ix2 - ix1) * (iy2 - iy1) * (iz2 - iz1)

def morton_code_3d(x, y, z):
    """
    简化的 Z-Order (Morton Code) 计算，用于兜底路由
    实际生产可用 pymorton 库，这里简单模拟
    """
    # 简单异或哈希替代，实际应使用位交叉
    return hash((x >> 5, y >> 5, z >> 5))