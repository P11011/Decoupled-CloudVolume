import zmq, os, sys
import numpy as np
import uuid
import traceback
import time
import msgpack
from multiprocessing import shared_memory

LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume'
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox # 推荐使用 Bbox 对象，更清晰

# =============================================================================
# 自动回收的 Numpy 数组 (RAII Pattern)
# =============================================================================
class AutoReleaseArray(np.ndarray):
    """
    继承自 numpy.ndarray。
    特性：
    1. 底层内存指向 SharedMemory (零拷贝)。
    2. 只有当这个数组对象被 Python GC 回收时，才会触发 shm.unlink()。
    """
    def __new__(cls, shape, dtype, shm_name, order='F'):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            raise RuntimeError(f"SharedMemory '{shm_name}' not found. Maybe created failed?")

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
        """
        [自动垃圾回收]
        当对象引用计数归零时执行。
        """
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
    def __init__(self, scheduler_addr, vol):
        """
        :param scheduler_addr: 调度器地址, e.g., "tcp://127.0.0.1:5555"
        :param meta_dtype: 数据类型，e.g., np.uint8
        :param background_color: 初始化填充值
        :param order: 内存布局, CloudVolume 常用 'F' (Fortran Order)
        """
        self.cv = vol
        self.client_id = f"{os.getpid()}_client_{uuid.uuid4().hex[:8]}"
        self.meta_dtype = np.dtype(vol.meta.data_type)
        self.background_color = vol.background_color
        self.num_channels = vol.meta.num_channels
        self.order = 'F'
        self.scheduler_addr = scheduler_addr
        # 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.client_id.encode('utf-8')
        self.socket.connect(scheduler_addr)
        
        # 设置轮询器 (用于设置超时)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        print(f"[Client] Started {self.client_id}, connected to {scheduler_addr}")

    def __getitem__(self, slices):
        """
        用户入口: vol[0:100, 0:100, 0:10]
        """
        time_0 = time.perf_counter()
        bbox = Bbox.from_slices(slices)
        shape = list(bbox.size3()) + [ self.num_channels ]

        # 检查是否为空请求
        if np.prod(shape) == 0:
            raise ValueError(f"Requested empty shape: {shape}")

        time_1 = time.perf_counter()
        # 2. 申请并初始化共享内存 (相当于 np.full)
        # 注意：这里只拿到了 shm 对象的引用，但我们需要把它包装进 AutoReleaseArray 才能返回
        shm_name = self._init_shared_buffer(shape)
        time_2 = time.perf_counter()
        try:
            # 3. 发送请求给调度器
            req_id = f"{os.getpid()}_req_{uuid.uuid4().hex[:8]}"
            self._send_request(req_id, bbox, shm_name, shape)
            time_3 = time.perf_counter()
            # 4. 阻塞等待回包
            self._wait_response(req_id)
            time_4 = time.perf_counter()
            # 5. 构造自动回收数组并返回
            # 此时 Worker 已经写完数据退出了，Client 接管这块内存
            result_arr = AutoReleaseArray(
                shape=shape,
                dtype=self.meta_dtype,
                shm_name=shm_name,
                order=self.order
            )
            time_5 = time.perf_counter()
            print(f"Time_cost(ms) Prepare_Info = {(time_1-time_0)*1000}, Init_Buffer = {(time_2-time_1)*1000}, "
                f"Send_Request = {(time_3-time_2)*1000}, Wait_Eesponse = {(time_4-time_3)*1000}, Construct_Array = {(time_5-time_4)*1000}")
            return result_arr

        except Exception as e:
            # 如果中间出错，必须手动清理 shm，因为 result_arr 还没创建成功
            # 无法依赖 __del__ 自动清理
            self._manual_cleanup(shm_name)
            raise e

    def _init_shared_buffer(self, shape):
        """
        即时申请共享内存，并执行 fill (np.full 逻辑)
        返回: shm_name
        """
        nbytes = int(np.prod(shape) * self.meta_dtype.itemsize)
        unique_name = f"{os.getpid()}_shm_{uuid.uuid4().hex}"
        
        # 申请 (create=True)
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=unique_name)
        
        try:
            # 填充初始值
            # 创建临时视图用于填充
            arr = np.ndarray(shape, dtype=self.meta_dtype, buffer=shm.buf, order=self.order)
            arr.fill(self.background_color)
            
            # 关闭本地句柄 (因为我们要把 name 传出去，稍后由 AutoReleaseArray 重新打开)
            # 注意：这里不能 unlink！
            shm.close() 
            return unique_name
        except Exception as e:
            shm.close()
            shm.unlink() # 出错要清理
            raise e

    def _send_request(self, req_id, bbox, shm_name, shape):
        """
        打包协议发送
        """
        bbox_list = [int(c) for c in bbox.to_list()]
        shape_list = [int(s) for s in shape]

        payload = {
            "type": "READ",
            "req_id": req_id,
            "bbox": bbox_list,       
            "shape": shape_list,     # 需要传给 Worker，因为它也需要知道怎么 reshape
            "dtype": str(self.meta_dtype),
            "shm_name": shm_name,
            "order": self.order
        }
        # 使用 msgpack 序列化
        self.socket.send(msgpack.packb(payload))

    def _wait_response(self, req_id, timeout_ms=5000):
        """
        阻塞等待指定 req_id 的回复
        """
        start_time = time.time()
        while True:
            socks = dict(self.poller.poll(timeout_ms))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                msg = self.socket.recv_multipart()
                msg = msg[-1]
                resp = msgpack.unpackb(msg)
                
                if resp.get('req_id') == req_id:
                    if resp['status'] == 'OK':
                        return # 成功
                    else:
                        raise RuntimeError(f"Worker Error: {resp.get('error')}")
                else:
                    print(f"[Client] Warning: Mismatched req_id. Expected {req_id}, got {resp.get('req_id')}")

            if (time.time() - start_time) * 1000 > timeout_ms:
                raise TimeoutError(f"Request {req_id} timed out after {timeout_ms}ms")

    def _manual_cleanup(self, shm_name):
        """
        辅助函数：在初始化 AutoReleaseArray 失败时的手动清理
        """
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            print(f"[Client] Manual cleanup of {shm_name} executed.")
        except:
            pass

