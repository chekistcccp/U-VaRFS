from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


class Reservoir:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity=capacity; self.rng=np.random.default_rng(seed); self.seen=0; self.arr=None

    def add(self, x: torch.Tensor):
        x=x.detach().float().cpu().numpy()
        if self.arr is None:
            self.arr=np.empty((self.capacity,x.shape[1]),dtype=np.float32)
        for row in x:
            if self.seen < self.capacity:
                self.arr[self.seen]=row
            else:
                j=int(self.rng.integers(0,self.seen+1))
                if j < self.capacity:
                    self.arr[j]=row
            self.seen += 1

    def value(self):
        n=min(self.seen,self.capacity)
        return self.arr[:n].copy() if self.arr is not None else np.empty((0,0),np.float32)


class TorchCosineIndex:
    """Exact cosine 1-NN on CUDA using chunked GEMM.

    This avoids the previous q.cpu().numpy() -> faiss-cpu round trip, which left
    the GPU almost idle during evaluation. Memory banks are kept in fp16/bf16
    on GPU and queries are scored by Tensor-Core matrix multiplication.
    """
    def __init__(
        self,
        memory: np.ndarray,
        device: str = "auto",
        dtype: str = "fp16",
        query_chunk: int = 2048,
        memory_chunk: int = 0,
    ):
        if memory.ndim != 2 or len(memory)==0:
            raise ValueError("empty memory bank")
        if not torch.cuda.is_available():
            raise RuntimeError("TorchCosineIndex requires CUDA")

        if device == "auto":
            device = "cuda:0"
        self.device=torch.device(device)
        self.query_chunk=max(1,int(query_chunk))
        self.memory_chunk=max(0,int(memory_chunk))
        if dtype == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype=torch.bfloat16
        else:
            self.dtype=torch.float16

        mem=torch.from_numpy(memory.astype(np.float32,copy=False)).to(self.device,non_blocking=False)
        mem=F.normalize(mem,dim=1).to(self.dtype)
        self.memory=mem.contiguous()

    @torch.inference_mode()
    def score(self, q: torch.Tensor) -> np.ndarray:
        q=F.normalize(q.detach().float(),dim=1)
        if q.device != self.device:
            q=q.to(self.device,non_blocking=True)
        q=q.to(self.dtype)

        outs=[]
        for qs in range(0,q.shape[0],self.query_chunk):
            qe=min(qs+self.query_chunk,q.shape[0])
            qq=q[qs:qe]
            if self.memory_chunk and self.memory.shape[0] > self.memory_chunk:
                best=torch.full((qq.shape[0],),-1.0,device=self.device,dtype=torch.float32)
                for ms in range(0,self.memory.shape[0],self.memory_chunk):
                    me=min(ms+self.memory_chunk,self.memory.shape[0])
                    sim=(qq @ self.memory[ms:me].T).float().amax(dim=1)
                    best=torch.maximum(best,sim)
            else:
                best=(qq @ self.memory.T).float().amax(dim=1)
            outs.append((1.0-best.clamp(-1.0,1.0)).cpu())
        return torch.cat(outs).numpy()


class CosineIndex:
    """FAISS fallback. With faiss-cpu this path is CPU-bound."""
    def __init__(self, memory: np.ndarray, use_gpu: bool = True, exact: bool = False, nprobe: int = 16):
        import faiss
        if memory.ndim != 2 or len(memory)==0:
            raise ValueError("empty memory bank")
        self.faiss=faiss
        mem=memory.astype(np.float32,copy=True)
        faiss.normalize_L2(mem)
        d=mem.shape[1]
        if exact or len(mem)<2048:
            index=faiss.IndexFlatIP(d)
        else:
            nlist=max(16,min(1024,int(4*math.sqrt(len(mem)))))
            quant=faiss.IndexFlatIP(d)
            index=faiss.IndexIVFFlat(quant,d,nlist,faiss.METRIC_INNER_PRODUCT)
            index.train(mem)
            index.nprobe=min(nprobe,nlist)
        self.on_gpu=False
        if use_gpu and torch.cuda.is_available() and hasattr(faiss,"StandardGpuResources"):
            try:
                self._res=faiss.StandardGpuResources()
                index=faiss.index_cpu_to_gpu(self._res,0,index)
                self.on_gpu=True
            except Exception:
                pass
        index.add(mem)
        self.index=index

    def score(self,q:torch.Tensor)->np.ndarray:
        arr=q.detach().float().cpu().numpy().astype(np.float32,copy=False)
        self.faiss.normalize_L2(arr)
        sim,_=self.index.search(arr,1)
        return 1.0-sim[:,0]


def make_index(memory: np.ndarray, eval_cfg: dict):
    backend=str(eval_cfg.get("nn_backend","auto")).lower()
    if backend == "auto":
        backend = "torch_gpu" if torch.cuda.is_available() else "faiss"
    if backend == "torch_gpu":
        return TorchCosineIndex(
            memory,
            device=str(eval_cfg.get("score_device","auto")),
            dtype=str(eval_cfg.get("score_dtype","fp16")),
            query_chunk=int(eval_cfg.get("query_chunk",2048)),
            memory_chunk=int(eval_cfg.get("memory_chunk",0)),
        )
    return CosineIndex(
        memory,
        bool(eval_cfg.get("faiss_gpu",True)),
        bool(eval_cfg.get("faiss_exact",False)),
        int(eval_cfg.get("faiss_nprobe",16)),
    )
