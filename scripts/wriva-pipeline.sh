WRIVA_FUNCTIONS="767539018665.dkr.ecr.us-east-1.amazonaws.com/jhu-wriva_functions:1.0"
ARTI_IMAGE="767539018665.dkr.ecr.us-east-1.amazonaws.com/jhu-global_artifact_det:1.0"
TRANSIENT_DET_IMAGE="767539018665.dkr.ecr.us-east-1.amazonaws.com/jhu-transient_artficat_det:1.0"
REGISTRATION_IMAGE="767539018665.dkr.ecr.us-east-1.amazonaws.com/jhu-registration:1.0"
MANERF_IMAGE_T4="767539018665.dkr.ecr.us-east-1.amazonaws.com/jhu-recon:1.0"
WRIVA_METRICS="767539018665.dkr.ecr.us-east-1.amazonaws.com/wriva-metrics:1.18.0"
COLMAP_IMAGE="767539018665.dkr.ecr.us-east-1.amazonaws.com/colmap:latest"
COORD_TRANS="767539018665.dkr.ecr.us-east-1.amazonaws.com/coord-transform:latest"

SCENE=aaviss-challenge3-pipeline-v2/t01_v01_s00_r08_ImageDensity_A01_iPad_LTS
ROOT=/export/r18c/data/dli/wriva/$SCENE
OUTROOT=$ROOT/docker_output
DOCKER="docker run --gpus device=0 --rm -v $ROOT:$ROOT"

mkdir -p $OUTROOT

$DOCKER \
    ${WRIVA_FUNCTIONS} \
    /bin/bash -c "python3 scripts/downsample.py $ROOT/input/images $OUTROOT/images; chmod -R 777 $OUTROOT; echo done"

$DOCKER \
    ${REGISTRATION_IMAGE} \
    /bin/bash -c "python3 airflow_execution_wriva.py -i $OUTROOT/images -d $OUTROOT -o $OUTROOT -s $OUTROOT/sparse"

$DOCKER \
    ${COLMAP_IMAGE} \
    /bin/bash -c "colmap image_undistorter --image_path $OUTROOT/images --input_path $OUTROOT/sparse --output_path $OUTROOT/nerf --output_type COLMAP && colmap model_converter --input_path $OUTROOT/nerf/sparse --output_path $OUTROOT/nerf/sparse --output_type TXT"

$DOCKER \
    ${MANERF_IMAGE_T4} \
    /bin/bash -c "python3 colmap2nerf.py --text $OUTROOT/nerf/sparse --colmap_db $OUTROOT/nerf/sparse --images $OUTROOT/nerf/images --aabb_scale 16 --images_relative ./nerf/images --output_transform_path $OUTROOT/transforms.json"

$DOCKER \
    ${WRIVA_FUNCTIONS} \
    /bin/bash -c "python3 scripts/nerf2wriva.py --images $OUTROOT/nerf/images --aabb_scale 16 --nerf_formatted_json $OUTROOT/transforms.json --wriva_formatted_json $OUTROOT/wriva_formatted_reconstructed_metadata.json"

$DOCKER \
    ${COORD_TRANS} \
    /bin/bash -c "python3 world2wriva.py --input_metadata_dir $ROOT/reference/ta2_metadata --reference_metadata_dir $ROOT/reference/metadata --wriva_form_reconstructed_input_data $OUTROOT/wriva_formatted_reconstructed_metadata.json --rmse_metrics_file $OUTROOT/rsme_coordinates.json --transform_w2m_file $OUTROOT/transform_world2model.json --wriva_form_reference_data $OUTROOT/wriva_form_reference_metadata.json"

$DOCKER \
    ${WRIVA_FUNCTIONS} \
    /bin/bash -c "python3 scripts/wriva2nerf_curr_curr.py --aabb_scale 16 --wriva_formatted_reference_metadata $OUTROOT/wriva_form_reference_metadata.json --reference_images $ROOT/reference/images --reference_images_relative_path ./images --output_nerf_formatted $OUTROOT/reference/transforms.json; chmod -R 777 $OUTROOT; echo done"
