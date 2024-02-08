# R3_mix
task_name="mnli_mix"

data_split="0,0,10"
data_path="../../data/MNLI/mnli_mix_example.json"
echo $data_path

# for MNLI, SNLI, max_prompt_seq_len = max_answer_seq_len = 256
# for raceHigh, max_prompt_seq_len = 1024, max_answer_seq_len = 512
# for boardgame, max_prompt_seq_len = max_answer_seq_len = 1024
temperature=0.7
do_sample=1
num_train_epochs=5
kl_ctl=0.3
reward_last_token=1
max_prompt_seq_len=256
max_answer_seq_len=256
actor_zero_stage=2
critic_zero_stage=3
actor_learning_rate=2e-6
critic_learning_rate=1e-6

output_base="/your_output_model_dir/${task_name}/"
log_base=./${task_name}/
mkdir $log_base

per_device_generation_batch_size=4
per_device_training_batch_size=4
gradient_accumulation_steps=2

# base model
model_type="llama2base7b"

actor_model_name_or_path="/your_sft_model_path"
critic_model_name_or_path="/your_sft_model_path"

task_name="R3_test"
output_path=${output_base}${task_name}
data_output_path=${output_path}/data_files
log_path=${log_base}${task_name}.log
echo $output_path

mkdir -p $output_path

deepspeed \
    --master_port 39925 \
    --num_gpus 8 \
    main.py \
    --test_data_type mnli \
    --seed 1234 \
    --data_path $data_path \
    --data_split $data_split \
    --data_output_path $data_output_path \
    --actor_learning_rate $actor_learning_rate \
    --critic_learning_rate $critic_learning_rate \
    --max_prompt_seq_len $max_prompt_seq_len \
    --max_answer_seq_len $max_answer_seq_len \
    --print_answers \
    --per_device_generation_batch_size $per_device_generation_batch_size \
    --per_device_training_batch_size $per_device_training_batch_size \
    --temperature $temperature \
    --do_sample $do_sample \
    --kl_ctl $kl_ctl \
    --actor_model_name_or_path $actor_model_name_or_path \
    --reward_last_token $reward_last_token \
    --num_train_epochs $num_train_epochs \
    --critic_model_name_or_path  $critic_model_name_or_path \
    --actor_zero_stage $actor_zero_stage \
    --critic_zero_stage $critic_zero_stage \
    --num_padding_at_beginning 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --deepspeed  \
    --enable_hybrid_engine \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --actor_dropout 0.0 \
    --output_dir $output_path \
    >   $log_path 2>&1
