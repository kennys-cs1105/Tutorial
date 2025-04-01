# vLLM 代码库

*Created By KennyS*

---

## 模块

- Entrypoint (LLM, API server): 接口 入口
- Engine: 利用好llm_engine
- Scheduler: 打包
- KV cache manager: KV cache高效存储, 内存空间
    - Paged Attention: 切分 -> LMCache
- Worker: 各种硬件worker
    - worker_base
    - worker
- Evictor: 先进先出？ 一个来了, 另一个就需要走了
    - Prefix caching: 两个request有同样的前缀，只需要计算一次，后面的可以复用。会有遗忘性问题。很多时候不需要将整个文本都进行处理。还有相似性前缀匹配问题(CacheBlend)。(What if prefix doesn't match? What if prefix cache on another machine, sharing across nodes.)
- Model executor (Model runner)
- Modelling (Other models -> vLLM)
- Attention backend -> flash attention

## 分布式

1. 因为一卡放不下, 所以需要多卡。
2. 更高的利用硬件资源

### Feature

*Distributed inference*

    - Why distributed inference
        - Communication device:
            - nvlink
            - infinity band
            - rdma
        - Communication library: `vllm/distributed/device_communicators`
            - `PyNccl`
            - `shared memory`: os
            - `custom allreduce`: A kernel just for all reduce operation
            - `torch.distributed`