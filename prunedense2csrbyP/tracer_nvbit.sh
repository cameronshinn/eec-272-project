# Do not run in docker container
# Install nvbit
./../util/tracer_nvbit/install_nvbit.sh
make -C ../util/tracer_nvbit/
# Generate traces
export CUDA_VISIBLE_DEVICES=0
LD_PRELOAD=./../util/tracer_nvbit/tracer_tool/tracer_tool.so ./prunedense2csrbyP 100 100
# Trace postprocessing for gpgpu-sim input
./../util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist
