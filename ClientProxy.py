import zmq, os, sys
import numpy as np
import uuid
import time
import msgpack
from multiprocessing import shared_memory

LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume.lib import Bbox

# =============================================================================
# 自动回收的 Numpy 数组 (RAII Pattern)
# =============================================================================
class AutoReleaseArray(np.ndarray):
    def __new__(cls, shape, dtype, shm_name, order='F'):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            raise RuntimeError(f"SharedMemory '{shm_name}' not found.")

        obj = super().__new__(cls, shape, dtype=dtype, buffer=shm.buf, order=order)
        obj._shm = shm
        obj._shm_name = shm_name
        obj._owns_memory = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._shm = getattr(obj, '_shm', None)
        self._shm_name = getattr(obj, '_shm_name', None)
        self._owns_memory = False

    def __del__(self):
        try:
            if getattr(self, '_owns_memory', False) and hasattr(self, '_shm') and self._shm:
                self._shm.close()
                try: 
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"Error in AutoReleaseArray destructor: {e}")

# =============================================================================
# 客户端代理 (Client Proxy)
# =============================================================================
class ClientProxy:
    # 大于此的请求走云
    SHM_THRESHOLD = 2000*2000*2000

    def __init__(self, scheduler_addr, vol):
        self.cv = vol
        self.cv.cache_thread = 0
        self.cv.partial_decompress_parallel = 1
        self.client_id = f"{os.getpid()}_client_{uuid.uuid4().hex[:8]}"
        self.meta_dtype = np.dtype(vol.meta.data_type)
        self.background_color = vol.background_color
        self.num_channels = vol.meta.num_channels
        self.order = 'F'
        self.scheduler_addr = scheduler_addr
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.client_id.encode('utf-8')
        self.socket.connect(scheduler_addr)
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        print(f"[Client] Started {self.client_id}")

    def __getitem__(self, slices):
        time_0 = time.perf_counter()
        bbox = Bbox.from_slices(slices)
        shape = list(bbox.size3()) + [ self.num_channels ]
        req_size = int(np.prod(shape))
        req_size_bytes = int(req_size * self.meta_dtype.itemsize)

        if np.prod(shape) == 0:
            raise ValueError(f"Requested empty shape: {shape}")

        if req_size < self.SHM_THRESHOLD:
            return self.cv[bbox]
        
        shm_name = self._init_shared_buffer_raw(req_size_bytes)
        
        time_1 = time.perf_counter()
        try:
            req_id = f"{os.getpid()}_req_{uuid.uuid4().hex[:8]}"
            self._send_request(req_id, bbox, shm_name, shape, req_size_bytes)
            
            self._wait_response(req_id)
            time_2 = time.perf_counter()
            
            result_arr = AutoReleaseArray(
                shape=shape,
                dtype=self.meta_dtype,
                shm_name=shm_name,
                order=self.order
            )
            # print(f"Shm Request time:{(time_1-time_0)*1000}ms, _wait_response:{(time_2-time_1)*1000}ms, AutoReleaseArray construct:{(time.perf_counter()-time_2)*1000}ms")
            return result_arr

        except Exception as e:
            self._manual_cleanup(shm_name)
            raise e

    def _init_shared_buffer_raw(self, nbytes):

        unique_name = f"{os.getpid()}_shm_{uuid.uuid4().hex}"
        # shm = shared_memory.SharedMemory(create=True, size=int(nbytes), name=unique_name)
        # shm.close() # 关闭句柄，但不 unlink
        return unique_name

    def _send_request(self, req_id, bbox, shm_name, shape, size_bytes):
        bbox_list = [int(c) for c in bbox.to_list()]
        shape_list = [int(s) for s in shape]

        payload = {
            "type": "READ",
            "req_id": req_id,
            "bbox": bbox_list,        
            "shape": shape_list,
            "dtype": str(self.meta_dtype),
            "shm_name": shm_name,
            "order": self.order,
            "data_size": int(size_bytes),
            "bg_color": self.background_color
        }
        self.socket.send(msgpack.packb(payload))

    def _wait_response(self, req_id, timeout_ms= 1000 * 1000):
        start_time = time.time()
        while True:
            socks = dict(self.poller.poll(timeout_ms))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                msg = self.socket.recv_multipart()
                msg = msg[-1]
                resp = msgpack.unpackb(msg)
                
                if resp.get('req_id') == req_id:
                    if resp['status'] == 'OK':
                        return
                    else:
                        raise RuntimeError(f"Worker Error: {resp.get('error')}")
            
            if (time.time() - start_time) * 1000 > timeout_ms:
                raise TimeoutError(f"Request {req_id} timed out")

    def _manual_cleanup(self, shm_name):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except:
            pass