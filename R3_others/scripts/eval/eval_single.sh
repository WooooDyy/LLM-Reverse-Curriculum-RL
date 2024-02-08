python_file="eval_batch.py"
output_base="eval_mnli/"
log_base="logs_mnli/"
mkdir ${output_base}
mkdir ${log_base}
mode="zero_shot"

task_name="R3_test"
model_dir="/your_eval_model_path"
data_path="../../data/MNLI/mnli_test.json"

output_file="${output_base}${task_name}.txt"
log_file="${log_base}${task_name}.log"


echo "--------------Eval-----------------"
echo "${model_dir}"


CUDA_VISIBLE_DEVICES=0   python "${python_file}" \
    --model_name_or_path "${model_dir}" \
    --results_path "${output_file}" \
    --data_path "${data_path}" \
    --mode ${mode} \
    --batch_size 4 \
    > ${log_file} 2>&1

