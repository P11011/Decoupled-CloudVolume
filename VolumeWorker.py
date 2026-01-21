import zmq
import msgpack
import numpy as np
import traceback
import os
import sys
import ctypes
import time
from multiprocessing import shared_memory

LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume

# 编译命令提示:
# g++ -O3 -mavx2 -fopenmp -shared -fPIC -o libfastfill.so fast_fill.cpp

lib_path = os.path.abspath("./libfastfill.so")
_lib = None

try:
    _lib = ctypes.CDLL(lib_path)
    
    # --- 1. 配置 parallel_fill_u8 (原 avx2) ---
    # void parallel_fill_u8(unsigned char* data, size_t size_bytes, unsigned char value, int num_threads)
    try:
        _lib.parallel_fill_u8.argtypes = [
            ctypes.c_void_p,   # data pointer
            ctypes.c_size_t,   # size in bytes
            ctypes.c_ubyte,    # fill value (0-255)
            ctypes.c_int       # num threads
        ]
        _lib.parallel_fill_u8.restype = None
    except AttributeError:
        # 兼容旧版本函数名 (如果 C++ 没重新编译)
        if hasattr(_lib, 'parallel_fill_avx2'):
            _lib.parallel_fill_u8 = _lib.parallel_fill_avx2
            _lib.parallel_fill_u8.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ubyte, ctypes.c_int]
            _lib.parallel_fill_u8.restype = None

    # --- 2. 配置 parallel_fill_u64 (新增) ---
    # void parallel_fill_u64(uint64_t* data, size_t num_elements, uint64_t value, int num_threads)
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

class VolumeWorker:
    def __init__(self, scheduler_addr, worker_idx, parallel=4):
        self.scheduler_addr = scheduler_addr
        self.worker_id = f"worker_{worker_idx}_{os.getpid()}".encode('utf-8')
        self.parallel = parallel
        
        print(f"[{self.worker_id.decode()}] Init CloudVolume (Parallel={parallel})...")
        self.cv = CloudVolume(
            '/CX/neuro_tracking/fafb-ffn1', 
            mip=0,
            fill_missing=True,
            log_path="/dev/shm/222.log",
            cache=True,
            # lru_bytes=80 * 1024**2, 
            # partial_decompress_parallel=parallel 
        )
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.worker_id

    def run(self):
        try:
            self.socket.connect(self.scheduler_addr)
            # 发送 READY 时携带自身的并行能力信息
            ready_payload = {"type": "READY", "parallel": self.parallel}
            self.socket.send(msgpack.packb(ready_payload))
            
            while True:
                frames = self.socket.recv_multipart()
                req = msgpack.unpackb(frames[-1])
                self._process_request(req)
        except Exception:
            traceback.print_exc()

    def _process_request(self, req):
        time1=time.perf_counter()
        client_id = req['client_id']
        req_id = req['req_id']
        shm_name = req['shm_name']
        shape = req['shape']
        dtype_str = req['dtype']
        order = req['order']
        bbox_list = req['bbox']
        bg_color = req.get('bg_color', 0) # 获取需要填充的颜色

        try:
            # 1. 打开共享内存
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            
            # 防止 ResourceTracker 报错 (Python bug workaround)
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(existing_shm._name, 'shared_memory')
            except:
                pass
            time2=time.perf_counter()
            # 2. 高性能 C++ 填充 (替代 Client 的 arr.fill)
            self._fast_fill(existing_shm, shape, dtype_str, bg_color, order)
            time3=time.perf_counter()
            # 3. CloudVolume 下载 (并发由 CloudVolume 内部根据 parallel 参数控制)
            slices = [
                slice(bbox_list[0], bbox_list[3]),
                slice(bbox_list[1], bbox_list[4]),
                slice(bbox_list[2], bbox_list[5])
            ]
            
            # 绑定 buffer 并下载
            self.cv.renderbuffer = existing_shm.buf
            self.cv[slices[0], slices[1], slices[2]]
            time4=time.perf_counter()
            print(f"Prepare time {(time2-time1)*1000}ms, _fast_fill time {(time3-time2)*1000}ms, cv time {(time4-time3)*1000}ms")
            # 4. 完成，关闭句柄
            existing_shm.close()

            # 5. 回复
            resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "OK"
            }
            self.socket.send(msgpack.packb(resp))

        except Exception as e:
            err_resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "ERROR",
                "error": str(e)
            }
            self.socket.send(msgpack.packb(err_resp))
            traceback.print_exc()

    def _fast_fill(self, shm, shape, dtype_str, value, order ='F'):
        """
        使用 C 标准库 memset 进行极速填充。
        注意：memset 按字节填充，仅当 value 为 0 或对 uint8 有效。
        如果 dtype 复杂且 value != 0，回退到 NumPy 的 C 实现。
        """
        
        arr = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf, order=order)
        self.fast_fill(arr, value, num_threads= self.parallel * 2)


    def fast_fill(self, array: np.ndarray, value: int, num_threads: int = 4):
        """
        智能分发填充任务到 C++ AVX2 函数
        支持 uint8 和 uint64 的高性能填充
        """
        # 0. 库未加载，回退
        if _lib is None:
            print(f"warning, _lib not found, fast_fill use numpy.fill()")
            array.fill(value)
            return

        # 1. 连续性检查
        # 共享内存创建的 ndarray 默认是连续的，但保险起见检查一下
        if not (array.flags['C_CONTIGUOUS'] or array.flags['F_CONTIGUOUS']):
            print("Warning: Array not contiguous, fallback to numpy")
            array.fill(value)
            return

        data_ptr = array.ctypes.data
        
        # 2. 策略分发
        
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