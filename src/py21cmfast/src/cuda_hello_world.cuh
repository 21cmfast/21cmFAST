#ifndef _CUDA_HELLO_WORLD_CUH
#define _CUDA_HELLO_WORLD_CUH

#ifdef __cplusplus
extern "C"
{
#endif
    int call_cuda();
    void print_key_device_properties();
#ifdef __cplusplus
}
#endif

#endif // _CUDA_HELLO_WORLD_CUH