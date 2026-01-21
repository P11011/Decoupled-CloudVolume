import ctypes
import multiprocessing.shared_memory
import numpy as np
import time
import os

# --- 1. 加载 C++ 库 ---
lib_path = os.path.abspath("./libfastfill.so")
try:
    lib = ctypes.CDLL(lib_path)
except OSError:
    print("错误: 找不到 libfastfill.so，请先执行编译命令。")
    exit(1)

# --- 2. 配置 C++ 函数签名 ---

# void parallel_fill_u8(unsigned char* data, size_t size_bytes, unsigned char value, int num_threads)
lib.parallel_fill_u8.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_uint8,
    ctypes.c_int
]

# void parallel_fill_u64(uint64_t* data, size_t num_elements, uint64_t value, int num_threads)
lib.parallel_fill_u64.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t, # 注意：这里是元素数量，不是字节数
    ctypes.c_uint64,
    ctypes.c_int
]

def benchmark_fill(label, data_ptr, size_bytes, fill_func, fill_value, threads, is_u64=False):
    """
    通用基准测试函数
    """
    print(f"[-] Running: {label:<20} | Threads: {threads}")
    
    # 转换为对应类型的指针
    if is_u64:
        c_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint64))
        count = size_bytes // 8 # 元素个数
    else:
        c_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8))
        count = size_bytes      # 字节数

    # 计时
    start_t = time.perf_counter()
    fill_func(c_ptr, count, fill_value, threads)
    end_t = time.perf_counter()
    
    duration = end_t - start_t
    gb = size_bytes / (1024**3)
    bw = gb / duration
    
    print(f"    Time: {duration*1000:.2f} ms | BW: {bw:.2f} GB/s")
    return bw

def verify_data(arr, value):
    """简单验证数据"""
    if arr[0] != value or arr[-1] != value or arr[len(arr)//2] != value:
        return False
    return True

def main():
    # --- 配置 ---
    SHM_NAME = "test_shm_full_benchmark"
    SIZE_GB = 1
    SIZE_BYTES = int(SIZE_GB * 1024**3)
    THREADS = 8
    
    # 测试值
    VAL_U8 = 205         # 0xCD
    VAL_U64 = 1234567890123456789 # 一个大的 64 位整数

    results = []

    print(f"=== 开始基准测试 (Size: {SIZE_GB} GB) ===\n")

    # ==========================================
    # 场景 1: Shared Memory (共享内存)
    # ==========================================
    try:
        # 清理旧的
        try:
            multiprocessing.shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except: pass

        shm = multiprocessing.shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SIZE_BYTES)
        
        # 获取地址
        shm_addr = ctypes.addressof(ctypes.c_uint8.from_buffer(shm.buf))
        
        # 1.1 SHM Uint8 Fill
        np_shm_u8 = np.ndarray((SIZE_BYTES,), dtype=np.uint8, buffer=shm.buf)
        np_shm_u8.fill(0) # Reset
        bw = benchmark_fill("SHM Uint8", shm_addr, SIZE_BYTES, lib.parallel_fill_u8, VAL_U8, THREADS, False)
        if not verify_data(np_shm_u8, VAL_U8): print("    [Error] Verification failed!")
        results.append(("Shared Mem", "Uint8", bw))

        # 1.2 SHM Uint64 Fill
        # 重新映射为 uint64 view
        np_shm_u64 = np.ndarray((SIZE_BYTES // 8,), dtype=np.uint64, buffer=shm.buf)
        np_shm_u64.fill(0) # Reset
        bw = benchmark_fill("SHM Uint64", shm_addr, SIZE_BYTES, lib.parallel_fill_u64, VAL_U64, THREADS, True)
        if not verify_data(np_shm_u64, VAL_U64): print("    [Error] Verification failed!")
        results.append(("Shared Mem", "Uint64", bw))

        shm.close()
        shm.unlink()

    except Exception as e:
        print(f"SHM Test Failed: {e}")

    print("-" * 40)

    # ==========================================
    # 场景 2: Local Memory (本地堆内存)
    # ==========================================
    try:
        # 使用 numpy 分配对齐的内存 (malloc)
        local_arr_u8 = np.empty((SIZE_BYTES,), dtype=np.uint8)
        local_addr = local_arr_u8.ctypes.data
        
        # 2.1 Local Uint8 Fill
        local_arr_u8.fill(0)
        bw = benchmark_fill("Local Uint8", local_addr, SIZE_BYTES, lib.parallel_fill_u8, VAL_U8, THREADS, False)
        if not verify_data(local_arr_u8, VAL_U8): print("    [Error] Verification failed!")
        results.append(("Local RAM", "Uint8", bw))

        # 2.2 Local Uint64 Fill
        # 重新 view 为 uint64
        local_arr_u64 = local_arr_u8.view(dtype=np.uint64)
        local_arr_u64.fill(0)
        bw = benchmark_fill("Local Uint64", local_addr, SIZE_BYTES, lib.parallel_fill_u64, VAL_U64, THREADS, True)
        if not verify_data(local_arr_u64, VAL_U64): print("    [Error] Verification failed!")
        results.append(("Local RAM", "Uint64", bw))

    except Exception as e:
        print(f"Local Test Failed: {e}")

    # ==========================================
    # 总结
    # ==========================================
    print("\n=== 测试总结 ===")
    print(f"{'Memory Type':<15} | {'Data Type':<10} | {'Bandwidth (GB/s)':<15}")
    print("-" * 45)
    for mem, dtype, bw in results:
        print(f"{mem:<15} | {dtype:<10} | {bw:.2f}")

if __name__ == "__main__":
    main()