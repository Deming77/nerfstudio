export DATA_ROOT=/hdd/data

for SCENE in \
    /hdd/data/aaviss-challenge3/* \
; do

echo ======= $SCENE ========

IMAGE_PATH=$SCENE/ta2_reference_images
COLMAP_PATH=$SCENE/ta2_colmap

cp $SCENE/ta2_reference_metadata/*.json $IMAGE_PATH

pushd /code/wriva/wrivalib
export PYTHONPATH=/code/wriva/wrivalib
python ./wrivalib/metadata/wriva_to_colmap.py \
    --root_dir $SCENE
popd
export PYTHONPATH=

rm $IMAGE_PATH/*.json

colmap feature_extractor \
    --SiftExtraction.use_gpu 0 \
    --database_path $COLMAP_PATH/sparse/0/database.db \
    --image_path $IMAGE_PATH

colmap exhaustive_matcher \
    --SiftMatching.use_gpu 0 \
    --database_path $COLMAP_PATH/sparse/0/database.db

mkdir -p $COLMAP_PATH/sparse/1 $COLMAP_PATH/sparse/2 $COLMAP_PATH/sparse/3

colmap model_converter \
    --input_path $COLMAP_PATH/sparse/0 \
    --output_path $COLMAP_PATH/sparse/1 \
    --output_type TXT

python ./scripts/aaviss_to_pts.py $SCENE

colmap point_triangulator \
    --database_path $COLMAP_PATH/sparse/0/database.db \
    --image_path $IMAGE_PATH \
    --input_path $COLMAP_PATH/sparse/2 \
    --output_path $COLMAP_PATH/sparse/3

python ./tools/points_from_wriva.py $SCENE --path ta2_colmap/sparse/3/points3D.bin -m colmap

done
