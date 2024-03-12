// Create cublas handle
cublasCreate(&handle);
return handle;
},
[](cublasHandle_t* handle)
{
    // Destroy cublas handle
    cublasDestroy(handle);
});
return creator();
}

std::shared_ptr<ncclComm_t> getNcclComm()
{
    static PerCudaCtxSingletonCreator<ncclComm_t> creator(
        []() -> auto
        {
            auto comm = std::unique_ptr<ncclComm_t>(new ncclComm_t);



<|im_end|>


<|im_start|>system
// Create nccl reduce plan
ncclReducePlanCreate(&reducePlan, comm.get(), *dtype, deviceIds.size(), deviceIds.data(), *redOp, rootId);
return reducePlan;


},
[](ncclReducePlan_t* reducePlan)
{
    // Destroy nccl reduce plan
    ncclReducePlanDestroy(reducePlan);
});
return creator();
}

std::shared_ptr<ncclGroupStart_t> getNcclGroupStart()
{
    static PerCudaCtxSingletonCreator<ncclGroupStart_t> creator(
        []() -> auto
        {
            auto groupStart = std::unique_ptr<ncclGroupStart_t>(new ncclGroupStart_t);



<|im_end|>


<|im_start|>system
// Create nccl group end
ncclGroupEndCreate(&groupEnd);
return groupEnd;


},
[](ncclGroupEnd_t* groupEnd)
{
    // Destroy nccl group end
    ncclGroupEndDestroy(groupEnd);
});
return creator();
}

std::shared_ptr<ncclGroupEnd_t> getNcclGroupEnd()
{
    static PerCudaCtxSingletonCreator<ncclGroupEnd_t> creator(
        []() -> auto
        {
            auto groupEnd = std::unique_ptr<ncclGroupEnd_t>(new ncclGroupEnd_t);




