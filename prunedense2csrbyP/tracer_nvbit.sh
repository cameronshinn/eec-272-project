#!/bin/sh

# Do not run in docker container
# Install nvbit
./../util/tracer_nvbit/install_nvbit.sh
make -C ../util/tracer_nvbit/

# Make directories for workload traces
sparsities="50 80 95"
layers="bottleneck_1_block_group2_1_1 bottleneck_1_block_group_projection_block_group3 bottleneck_3_block_group1_1_1 bottleneck_projection_block_group_projection_block_group3"
ogdir=$(pwd)

for sp in $sparsities
do
    for layer in $layers
    do
        # Make new dir for layer traces and move into it
        tracedir="$(pwd)/rn50_traces/$sp/$layer/"
        mkdir -p $tracedir
        echo "Moving into $tracedir"
        cd $tracedir

        smtx_path=$ogdir/../rn50_magnitude_pruning/$sp/$layer.smtx

        # Generate traces
        export CUDA_VISIBLE_DEVICES=0
        LD_PRELOAD=$ogdir/../util/tracer_nvbit/tracer_tool/tracer_tool.so $ogdir/prunedense2csrbyP $smtx_path $sp

        # Trace postprocessing for gpgpu-sim input
        $ogdir/../util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist

        # Move back to original directory
        echo "Moving back to $ogdir"
        cd $ogdir
    done
done
