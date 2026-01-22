import ctypes
import multiprocessing.shared_memory
import numpy as np
import time
import os
import sys

# --- 1. 加载 C++ 库 ---
# 编译命令: g++ -O3 -mavx2 -fopenmp -fPIC -shared fast_fill.cpp -o libfastfill.so
lib_path = os.path.abspath("./libfastfill.so")
try:
    lib = ctypes.CDLL(lib_path)
except OSError:
    print(f"错误: 无法加载 {lib_path}")
    sys.exit(1)

# --- 配置 C++ 函数 ---
try:
    lib.parallel_fill_u8.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int]
    lib.parallel_fill_u8.restype = None
    
    lib.parallel_fill_u64.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int]
    lib.parallel_fill_u64.restype = None
except AttributeError:
    print("错误: C++ 库函数签名不匹配，请重新编译库文件。")
    sys.exit(1)

def run_perf_test(label, func, *args):
    """通用性能计时器"""
    print(f"[-] Running: {label:<30} ...", end="", flush=True)
    start_t = time.perf_counter()
    func(*args)
    end_t = time.perf_counter()
    duration = end_t - start_t
    print(f" Done in {duration*1000:.2f} ms")
    return duration

def benchmark_suite(mem_type_name, array, threads, results_list):
    """
    对同一块内存先后执行：FastFill (C++) -> Standard Fill (NumPy)
    """
    size_bytes = array.nbytes
    size_gb = size_bytes / (1024**3)
    data_ptr = array.ctypes.data
    
    # -------------------------------------------------
    # 测试 1: C++ FastFill (Cold/Warm Write)
    # 这是第一次接触这块内存(如果是新分配的)，会包含 Page Fault 开销
    # -------------------------------------------------
    if array.dtype == np.uint8:
        c_func = lib.parallel_fill_u8
        c_val = 205
        count = size_bytes
        ptr_cast = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8))
        dtype_name = "Uint8"
    else:
        c_func = lib.parallel_fill_u64
        c_val = 1234567890123456789
        count = size_bytes // 8 # 元素个数
        ptr_cast = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint64))
        dtype_name = "Uint64"

    # 执行 C++ 填充
    def _cpp_wrapper():
        c_func(ptr_cast, count, c_val, threads)
    
    t_fast = run_perf_test(f"{mem_type_name} {dtype_name} [FastFill]", _cpp_wrapper)
    bw_fast = size_gb / t_fast

    # -------------------------------------------------
    # 测试 2: Standard NumPy Fill (Warm Write)
    # 在 FastFill 之后立即执行。此时内存物理页已分配(Warm)，
    # 测试的是 Python/NumPy 的单线程/多线程内存写入能力。
    # -------------------------------------------------
    def _numpy_wrapper():
        array.fill(0) # 填回 0

    t_std = run_perf_test(f"{mem_type_name} {dtype_name} [NumPy Fill]", _numpy_wrapper)
    bw_std = size_gb / t_std

    # 记录结果
    results_list.append({
        "Memory": mem_type_name,
        "Type": dtype_name,
        "FastFill BW": bw_fast,
        "NumPy BW": bw_std,
        "Speedup": bw_fast / bw_std
    })

def main():
    # 配置
    SHM_NAME = "test_shm_rw_bench"
    SIZE_GB = 1
    SIZE_BYTES = int(SIZE_GB * 1024**3)
    THREADS = 24
    
    results = []

    print(f"=== 内存写入基准测试 (Size: {SIZE_GB} GB | Threads: {THREADS}) ===")
    print("提示: 建议使用 'numactl --interleave=all python test.py' 运行\n")

    # ==========================================
    # 场景 1: Shared Memory
    # ==========================================
    try:
        try: multiprocessing.shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except: pass

        print("--- Shared Memory (/dev/shm) ---")
        shm = multiprocessing.shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SIZE_BYTES)
        
        # 测试 Uint8
        # arr_u8 = np.ndarray((SIZE_BYTES,), dtype=np.uint8, buffer=shm.buf)
        # benchmark_suite("Shared Mem", arr_u8, THREADS, results)

        # 测试 Uint64
        arr_u64 = np.ndarray((SIZE_BYTES // 8,), dtype=np.uint64, buffer=shm.buf)
        benchmark_suite("Shared Mem", arr_u64, THREADS, results)

        shm.close()
        shm.unlink()

    except Exception as e:
        print(f"SHM Test Failed: {e}")

    print("\n" + "-"*40 + "\n")

    # ==========================================
    # 场景 2: Local Memory
    # ==========================================
    try:
        print("--- Local Memory (malloc) ---")
        # # 测试 Uint8
        local_u8 = np.empty((SIZE_BYTES,), dtype=np.uint8)
        # benchmark_suite("Local RAM", local_u8, THREADS, results)

        # 测试 Uint64 (View)
        local_u64 = local_u8.view(dtype=np.uint64)
        benchmark_suite("Local RAM", local_u64, THREADS, results)

    except Exception as e:
        print(f"Local Test Failed: {e}")

    # ==========================================
    # 结果汇总表
    # ==========================================
    print("\n=== 最终对比结果 ===")
    header = f"{'Memory':<12} | {'Type':<8} | {'FastFill (GB/s)':<18} | {'NumPy (GB/s)':<15} | {'Speedup':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        print(f"{r['Memory']:<12} | {r['Type']:<8} | "
              f"{r['FastFill BW']:<18.2f} | {r['NumPy BW']:<15.2f} | "
              f"{r['Speedup']:<8.2f}x")

if __name__ == "__main__":
    main()