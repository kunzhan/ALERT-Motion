#!/bin/bash


# 定义要执行的命令
command_to_run="python run_mld.py"

# 定义参数列表
parameters=()

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
# current_time="2024-06-11_18-08-24"
# 定义运行总数
all_count=1

# 定义并行数量
parallel_count=1


total_rounds=$(expr $all_count / $parallel_count )

# 使用循环定义参数列表
# 定义并行数量
i=0
while (( $i < $parallel_count )); do
    parameters1+=("$i");
    ((i=$i+1));
done
# 定义总轮数
i=0
while (( $i < $total_rounds )); do
    parameters2+=("$i");
    ((i=$i+1));
done
# parameters2+=("4")

# 使用循环执行命令，为每次执行提供不同的参数
for param2 in "${parameters2[@]}"; do
    for param1 in "${parameters1[@]}"; do
        param=$(expr $param1 + \( $param2 \) \* $parallel_count )
        $command_to_run --i "$param" --v_str $current_time &
        # echo "$param $current_time"
    done
    wait  # 等待并行命令执行完毕
done