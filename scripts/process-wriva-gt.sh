export DATA_ROOT=/export/r18c/data/dli/wriva

for SCENE in \
    $DATA_ROOT/aaviss-challenge3-pipeline-v2/* \
; do

echo ======= $SCENE ========

# python ./tools/process_wriva_set.py $SCENE

for split in images ta2_images; do

echo ------- $split -------

python ./tools/generate_seg_masks.py \
    $SCENE/reference/images_4/$split \
    $SCENE/seg_masks_obj/masks/$split \
    --visualization $SCENE/seg_masks_obj/visualization/$split

# python ./tools/generate_sky_masks.py \
#     $root/images_4/$split \
#     $root/sky_masks_xz/masks/$split \
#     --visualization $root/sky_masks_xz/visualization/$split

python ./tools/generate_sky_masks.py \
    $SCENE/reference/images_4/$split \
    $SCENE/sky_masks_dt/masks/$split \
    --visualization $SCENE/sky_masks_dt/visualization/$split \
    --method detectron

python ./tools/combine_masks.py \
    --image $SCENE/reference/images_4/$split \
    --output $SCENE/combined_masks/masks/$split \
    --visualization $SCENE/combined_masks/visualization/$split \
    $SCENE/seg_masks_obj/masks/$split \
    # $root/sky_masks_xz/masks/$split \
    $SCENE/sky_masks_dt/masks/$split

# python ./tools/combine_masks.py \
#     --image $root/images_4/$split \
#     --output $root/sky_masks/masks/$split \
#     --visualization $root/sky_masks/visualization/$split \
#     $root/sky_masks_xz/masks/$split \
#     $root/sky_masks_dt/masks/$split

done

done


# Generate ground-only, ground+drones, drone-only transforms

# for SCENE in \
#     $DATA_ROOT/aaviss-challenge3/*_ReconstructedArea_* \
# ; do

# echo ======= $SCENE ========

# python ./tools/reconarea_variations.py $SCENE

# done
