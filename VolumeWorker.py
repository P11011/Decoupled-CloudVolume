import zmq
import msgpack
import numpy as np
import traceback
import os
from multiprocessing import shared_memory

LOCAL_CLONE = '/CX/neuro_tracking/xinr/cloudvolume_test/Cloudvolume'
import os
import sys
if os.path.exists(LOCAL_CLONE):
    sys.path.insert(0, LOCAL_CLONE)

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox 

class VolumeWorker:
    def __init__(self, scheduler_addr, worker_idx):
        self.scheduler_addr = scheduler_addr
        self.worker_id = f"worker_{worker_idx}_{os.getpid()}".encode('utf-8')
        
        # 初始化 CloudVolume (开启 LRU 内存缓存)
        print(f"[{self.worker_id.decode()}] Initializing CloudVolume...")
        self.cv = CloudVolume(
            '/CX/neuro_tracking/fafb-ffn1', 
            mip=0,
            fill_missing=True,
            cache=True,
            log_path="/dev/shm/222.log",            #
            lru_bytes= 80* 1024**2, 
            partial_decompress_parallel=20
        )
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.worker_id

    def run(self):
        try:
            self.socket.connect(self.scheduler_addr)
            print(f"[{self.worker_id.decode()}] Connected. Sending READY.")
            
            # 发送上线信号
            ready_payload = {"type": "READY"}
            self.socket.send(msgpack.packb(ready_payload))
            
            while True:
                try:
                    frames = self.socket.recv_multipart()
                    msg = frames[-1]
                    req = msgpack.unpackb(msg)
                    
                    self._process_request(req)
                    
                except Exception as e:
                    print(f"[{self.worker_id.decode()}] Loop Error: {e}")
                    traceback.print_exc()
        except Exception as e:
            traceback.print_exc()

    def _process_request(self, req):
        client_id = req['client_id']
        req_id = req['req_id']
        bbox_list = req['bbox'] # Flat list
        shape = req['shape']
        shm_name = req['shm_name']
        dtype_str = req['dtype']
        order = req['order'] # 'F' or 'C'
        
        try:
            # 转换 BBox
            slices = [
                slice(bbox_list[0], bbox_list[3]),
                slice(bbox_list[1], bbox_list[4]),
                slice(bbox_list[2], bbox_list[5])
            ]
            
            # 写入共享内存 (Memcpy)
            self._write_to_shm(slices, shm_name, shape, dtype_str, order)
            
            # 发送成功回包
            resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "OK"
            }
            self.socket.send(msgpack.packb(resp))
            # print(f"[{self.worker_id.decode()}] Processed {req_id}")

        except Exception as e:
            print(f"[{self.worker_id.decode()}] Error processing {req_id}: {e}")
            # 发送错误回包
            err_resp = {
                "type": "RESULT",
                "client_id": client_id,
                "req_id": req_id,
                "status": "ERROR",
                "error": str(e)
            }
            self.socket.send(msgpack.packb(err_resp))

    def _write_to_shm(self, slices, shm_name, target_shape, dtype_str, order):
        """
        连接共享内存并写入数据
        """
        existing_shm = None
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            try:
                from multiprocessing import resource_tracker
                # 需要移除这块内存的追踪记录,由用户端释放，这边避免报资源泄漏错误
                resource_tracker.unregister(existing_shm._name, 'shared_memory')
            except Exception:
                pass

            self.cv.renderbuffer = existing_shm.buf
            self.cv[slices[0], slices[1], slices[2]]
        except Exception as e:
            traceback.print_exc()  
        finally:
            if existing_shm:
                existing_shm.close() # 只是关闭句柄，不销毁内存