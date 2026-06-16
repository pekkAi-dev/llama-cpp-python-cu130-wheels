# llama-cpp-python cu130 Linux wheels

Pre-built `linux_x86_64` wheels of [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) for CUDA 13.x. Single `py3-none` wheel — installs on Python 3.9 through 3.14 from one file.

Reason this exists: Installing something named cu131 when on cu130 hurts. 

Tested working (not just imported) on python 3.9-3.14 and cu130-cu133.

## Install

```bash
pip install https://github.com/pekkAi-dev/llama-cpp-python-cu130-wheels/releases/latest/download/llama_cpp_python-0.3.40+cu130-py3-none-linux_x86_64.whl
```
https://github.com/pekkAi-dev/llama-cpp-python-cu130-wheels/releases/download/0.3.40/llama_cpp_python-0.3.40+cu130-py3-none-linux_x86_64.whl
All release assets are the same file, just with "cu13x" being different in name for easy "TL;DR -> Install" (i do it too, sometimes) and "cu13" so one can save locally and name is what it covers.

Note: If on first import you get `libnccl.so.2: cannot open shared object file`, install NCCL from somewhere, these work:

```bash
pip install nvidia-nccl-cu13           # pip-only environments
conda install -c nvidia nccl           # conda environments
# or your distro's nccl/libnccl2 package
```

# The techical stuff

## Runtime requirements

- Linux x86_64
- Python 3.9+
- NVIDIA driver ≥ R580 (any CUDA 13.x driver)
- `libnccl.so.2` discoverable on the dynamic loader path
- libstdc++ ≥ GLIBCXX_3.4.30 (Ubuntu 22.04+ era distros)

## What's in the wheel

- **CUDA archs**: SM 75, 80, 86, 87, 89, 90, 100, 120, 121 (`-real`, no PTX) — Turing through Blackwell consumer & automotive
- **CPU variants**: x64, sse42, sandybridge, ivybridge, piledriver, haswell, skylakex, cannonlake, cascadelake, cooperlake, icelake, alderlake, sapphirerapids, zen4 — `GGML_CPU_ALL_VARIANTS`, auto-selected at runtime
- **Multimodal**: `libmtmd.so` included (vision + audio for supported chat handlers)
- **Not built**: server, UI, examples, tests, command-line tools

## Build configuration

- CUDA toolkit 13.0.88 (`conda install -c nvidia cuda-toolkit=13.0`)
- Host compiler: `x86_64-conda-linux-gnu-{gcc,g++}` 14.x, sysroot-isolated
- `GGML_CUDA=ON GGML_NATIVE=OFF`
- `GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON GGML_OPENMP=ON`
- `GGML_CUDA_FORCE_MMQ=ON CUDA_SEPARABLE_COMPILATION=ON`
- Wheel retagged from `cp314-cp314` to `py3-none` (pure ctypes binding, no CPython ABI surface)

## Notes

- `llama_supports_gpu_offload()` returns `False` even when offload works — quirk of `GGML_BACKEND_DL=ON` builds. Confirm GPU offload by loading a model with `verbose=True` and looking for `register_backend: registered backend CUDA` and `load_tensors: offloaded N/N layers to GPU`.
- The `+cu130` local version segment is informational. The wheel runs on any CUDA 13.x runtime (cu130, cu131, cu132, cu133) since llama.cpp dynamically links `libcudart.so.13`.

## License

MIT, inherited from [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) upstreams. This release ships binaries only; all source belongs to the upstream projects.
