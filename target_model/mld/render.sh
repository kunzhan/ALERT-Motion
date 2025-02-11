#!/bin/bash

ids=("0" "1" "2" "3" "4" "5" "6" "7")
# models=("MDM" "MLD")
id_str=("008900" "000565" "012568" "013898" "M000903" "M008668" "M014109" "M001577")
devices=("0" "1" "2" "3" "4" "5" "6" "7")
# methods=("target" "baseline" "llm" "mac_prompt")


# models=("mld")
# methods=("target" "llm")
# ids=("0" "1" "2" "3")
# devices=("6" "6" "7" "7")
# i=("0" "1")
# $(expr $item2 + 3 )
# for item1 in "${models[@]}"; do
#     for item2 in "${ids[@]}"; do
#         {
#         save_dir="target_model/mld/render_file/$item1/${methods[$item2]}"
#         blender/2.93/python/bin/python3.9 -m fit --dir $save_dir --save_folder $save_dir --gpu_ids ${devices[$item2]}
#         # blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=video --joint_type=HumanML3D
#         # blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=sequence --joint_type=HumanML3D
#         }&
#     done
#     wait
# done

# for item1 in "${models[@]}"; do
#     for item2 in "${ids[@]}"; do
#         {
#         save_dir="target_model/mld/render_file/$item1/${methods[$item2]}"
#         # blender/2.93/python/bin/python3.9 -m fit --dir $save_dir --save_folder $save_dir --gpu_ids ${methods[$item2]}
#         blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=video --joint_type=HumanML3D
#         # blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=sequence --joint_type=HumanML3D
#         }&
#     done
#     wait
# done

for item1 in "${ids[@]}"; do
{
    file="target_model/mld/MM/MDM/${id_str[$item1]}.npy"
    save_dir="target_model/mld/MM/MDM/"
    blender/2.93/python/bin/python3.9 -m fit --files $file --save_folder $save_dir --gpu_ids ${devices[$item1]}
    # blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=video --joint_type=HumanML3D
    blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$save_dir --mode=sequence --joint_type=HumanML3D

    # blender/blender --background --python render.py -- --cfg=./configs/render.yaml --dir="target_model/mld/render_file" --mode=sequence --joint_type=HumanML3D
    }&
done