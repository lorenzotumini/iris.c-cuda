# CUDA optimization candidates

Measured on an RTX 3070 Ti 8 GiB with Z-Image Turbo BF16 at 1024 x 1024,
8 Euler steps: Iris used 3.45 GiB peak board memory and spent 37.05 seconds in
denoising. The low peak leaves room to trade VRAM for speed.

## Highest-value experiments

1. Add an adjustable/adaptive Z-Image BF16 weight cache. The current CUDA path
   invalidates every block after use, so the 11.5 GiB transformer may cross PCIe
   again on every denoising step. Test 0, 1.5, 2.5, and 3.5 GiB cache budgets,
   reserving enough memory for resolution-dependent activations and VAE decode.
2. Prefetch block N+1 on a separate transfer stream while block N computes.
   Use two reusable device weight arenas so allocation and eviction do not
   serialize the pipeline.
3. Stage mmap-backed weights through reusable pinned host buffers. Measure
   host-to-device bandwidth and overlap before and after; avoid pinning the
   entire checkpoint on low-RAM machines.
4. Keep small refiners and other weights reused on every step resident first.
   Release the transformer cache before VAE decode.
5. Add a persistent benchmark path that reuses the model context and cached
   prompt embedding across seeds. Keep one-shot and persistent timings separate.

After transfer stalls are reduced, profile attention, normalization, layout
conversion, and small-kernel launch gaps with Nsight Systems. Consider further
fusion or cuBLASLt algorithm tuning only where the profile shows a substantial
remaining cost.

## Comparison caveat

ComfyUI's 12.29-second warm Z-Image result reused a persistent process, model
objects, and the same prompt embedding. Its first complete workflow took 26.20
seconds. ComfyUI also uses pinned host memory and two asynchronous weight
offload streams. Compare Iris against both modes and include changed-prompt
runs before attributing the gap to CUDA kernels alone.

Apple Silicon does not pay the same discrete-GPU transfer cost: upstream Iris
is deliberately memory efficient, but Metal can access shared unified memory.
The CUDA cache and prefetch policy should therefore remain backend-specific.
