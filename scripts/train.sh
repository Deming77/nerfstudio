export DATA_ROOT=/export/r18c/data/dli/wriva

for scene in \
    aaviss-challenge3-pipeline-v2/t01_v01_s00_r08_ImageDensity_A01_iPad_LTS \
; do

ns-train gss \
    --data $DATA_ROOT/$scene/docker_output/transforms_full.json \
    --vis tensorboard \
    --output-dir $DATA_ROOT/$scene/output/nerfstudio \
    --experiment-name $scene \
    --pipeline.model.npz_file $DATA_ROOT/$scene/docker_output/sfm_points.npz \
    --pipeline.model.densify_until_iter 50000 \
    --max_num_iterations 100000 \
    nerfstudio-data \
    # --mask-root $DATA_ROOT/$scene/docker_output/seg_masks_obj/masks

done

# export DATA_ROOT=/export/r18c/data/dli/wriva

# for scene in \
#     aaviss-challenge3-pipeline-v2/t03_v01_s03_r06_ReconstructedArea_A01 \
# ; do

# ns-train gss-big \
#     --data $DATA_ROOT/$scene/transforms_ground+drones.json \
#     --output-dir /output/nerfstudio/train \
#     --experiment-name $scene/gt \
#     --suffix big-skyzeros-withdrones \
#     --pipeline.model.npz_file $DATA_ROOT/$scene/sfm_points.npz \
#     nerfstudio-data \
#     --downscale-factor 4 \
#     --mask-root $DATA_ROOT/$scene/combined_masks/masks \
#     --special-mask-roots sky_mask:$DATA_ROOT/$scene/sky_masks/masks

# ns-train gss-big \
#     --data $DATA_ROOT/$scene/transforms_ground+drones.json \
#     --output-dir /output/nerfstudio/train \
#     --experiment-name $scene/gt \
#     --suffix big-di1000-skyzeros-withdrones \
#     --pipeline.model.npz_file $DATA_ROOT/$scene/sfm_points.npz \
#     --pipeline.model.densification_interval 1000 \
#     nerfstudio-data \
#     --downscale-factor 4 \
#     --mask-root $DATA_ROOT/$scene/combined_masks/masks \
#     --special-mask-roots sky_mask:$DATA_ROOT/$scene/sky_masks/masks

# ns-train gss-gfeats-hashgrid \
#     --data $DATA_ROOT/$scene/transforms_ground+drones.json \
#     --output-dir /output/nerfstudio/train \
#     --experiment-name $scene/gt \
#     --suffix big-skyzeros-withdrones-gfeats \
#     --pipeline.model.renderer gss \
#     --pipeline.model.npz_file $DATA_ROOT/$scene/sfm_points.npz \
#     --max_num_iterations 100000 \
#     --pipeline.model.densify_until_iter 50000 \
#     nerfstudio-data \
#     --downscale-factor 4 \
#     --mask-root $DATA_ROOT/$scene/combined_masks/masks \
#     --special-mask-roots sky_mask:$DATA_ROOT/$scene/sky_masks/masks

# ns-train gss-gfeats-hashgrid \
#     --data $DATA_ROOT/$scene/transforms_ground+drones.json \
#     --output-dir /output/nerfstudio/train \
#     --experiment-name $scene/gt \
#     --suffix big-di1000-skyzeros-withdrones-gfeats \
#     --pipeline.model.renderer gss \
#     --pipeline.model.npz_file $DATA_ROOT/$scene/sfm_points.npz \
#     --pipeline.model.densification_interval 1000 \
#     --max_num_iterations 100000 \
#     --pipeline.model.densify_until_iter 50000 \
#     nerfstudio-data \
#     --downscale-factor 4 \
#     --mask-root $DATA_ROOT/$scene/combined_masks/masks \
#     --special-mask-roots sky_mask:$DATA_ROOT/$scene/sky_masks/masks

# done
