import numpy as np
import ctypes, os
import ctypes.util
# 编译命令提示:
# g++ -O3 -mavx2 -fopenmp -shared -fPIC -o libfastfill.so fast_fill.cpp

lib_path = os.path.abspath("./libfastfill.so")
_lib = None

try:
    _lib = ctypes.CDLL(lib_path)
    
    try:
        _lib.parallel_fill_u8.argtypes = [
            ctypes.c_void_p,   # data pointer
            ctypes.c_size_t,   # size in bytes
            ctypes.c_ubyte,    # fill value (0-255)
            ctypes.c_int       # num threads
        ]
        _lib.parallel_fill_u8.restype = None
    except AttributeError:
        if hasattr(_lib, 'parallel_fill_avx2'):
            _lib.parallel_fill_u8 = _lib.parallel_fill_avx2
            _lib.parallel_fill_u8.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ubyte, ctypes.c_int]
            _lib.parallel_fill_u8.restype = None
    try:
        _lib.parallel_fill_u64.argtypes = [
            ctypes.c_void_p,   # data pointer
            ctypes.c_size_t,   # num elements (注意：不是字节数！)
            ctypes.c_uint64,   # fill value (64-bit)
            ctypes.c_int       # num threads
        ]
        _lib.parallel_fill_u64.restype = None
    except AttributeError:
        print("Warning: parallel_fill_u64 not found in .so. Please recompile C++ code.")

except OSError:
    print(f"Warning: Could not load {lib_path}, fast fill will not be available.")
    _lib = None

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

def _fast_fill(buffer, shape, dtype_str, parallel, value, order ='F'):
    """
    使用 C 标准库 memset 进行极速填充。
    注意：memset 按字节填充，仅当 value 为 0 或对 uint8 有效。
    如果 dtype 复杂且 value != 0，回退到 NumPy 的 C 实现。
    """
    
    arr = np.ndarray(shape, dtype=dtype_str, buffer=buffer, order=order)
    fast_fill(arr, value, num_threads= parallel)


def fast_fill(array: np.ndarray, value: int, num_threads: int = 4):
    """
    智能分发填充任务到 C++ AVX2 函数
    支持 uint8 和 uint64 的高性能填充
    """
    #  库未加载，回退
    if _lib is None:
        print(f"warning, _lib not found, fast_fill use numpy.fill()")
        array.fill(value)
        return

    # 连续性检查
    if not (array.flags['C_CONTIGUOUS'] or array.flags['F_CONTIGUOUS']):
        print("Warning: Array not contiguous, fallback to numpy")
        array.fill(value)
        return

    data_ptr = array.ctypes.data
    
    # 策略分发
    
    # 策略 A: 如果 value 是 0，所有类型都等价于 memset 0
    # 这是最高效的路径，支持 float, int32, int64 等所有类型
    if value == 0:
        _lib.parallel_fill_u8(data_ptr, array.nbytes, 0, num_threads)
        return

    # 策略 B: 64位整数 (uint64, int64)
    # 只有当 value != 0 时才需要专门处理类型
    if array.itemsize == 8 and np.issubdtype(array.dtype, np.integer):
        # C++ 接口要求 num_elements，不是 nbytes
        num_elements = array.size
        # 强制转换为 c_uint64 (处理 int64 负数情况)
        c_val = ctypes.c_uint64(int(value))
        
        if hasattr(_lib, 'parallel_fill_u64'):
            _lib.parallel_fill_u64(data_ptr, num_elements, c_val, num_threads)
        else:
            array.fill(value) # 没编译新库则回退
        return

    # 策略 C: 8位整数 (uint8, int8)
    if array.itemsize == 1:
        val_byte = int(value) & 0xFF
        _lib.parallel_fill_u8(data_ptr, array.nbytes, val_byte, num_threads)
        return

    # 策略 D: 其他情况 (如 uint32=0x1234, float=1.5)
    # 目前 C++ 只实现了 u8 和 u64，其他类型回退到 Numpy
    print(f"warning, fast_fill use numpy.fill()")
    array.fill(value)
    