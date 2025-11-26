GPU_LIST=('0' '1' '2' '3' '4')
PID_LIST=()
FREE_GPUS=()
NUM_GPUS_PER_TASK=1

for i in ${GPU_LIST[*]};
do
    PID_LIST+=('nuLL')
done
echo GPU=${GPU_LIST[*]}


LOG_FOLDER_PATH="outputs/logs"
if [ ! -d "$LOG_FOLDER_PATH" ]; then
    mkdir -p "$LOG_FOLDER_PATH"
fi


pid_exist(){
    input_id=$1
    key_word='run_dp.sh'
    proc_num=$(ps -aux | grep $input_id | grep $key_word | grep -v grep | wc -l)
    return $proc_num
}


gpu_monitor(){
    FREE_GPUS=()
    for ((idx=0; idx<${#GPU_LIST[*]}; idx++));
    do
        temp_gpu=${GPU_LIST[${idx}]}
        temp_pid=${PID_LIST[${idx}]}
        pid_exist $temp_pid
        p_num=$?
        if [ $p_num -eq 0 ]; then
            FREE_GPUS+=($temp_gpu)
            if [ ${#FREE_GPUS[*]} -eq $NUM_GPUS_PER_TASK ]; then
                return 1
            fi
        fi
    done
    sleep 120s
    return 0
}


date
echo ------------------- start training ------------------------
for AT in 'tandem';
do
    for CFG in 'configs/train_qwen2_7b_udewt_dp.json' 'configs/train_gemma2_9b_udewt_dp.json' 'configs/train_llama3_1_8b_udewt_dp.json';
    do
        while true
        do
            gpu_monitor
            have_free_gpu=$?
            if [ $have_free_gpu -eq 1 ]
            then
                LOG_FILE=outputs/logs/CFG_${CFG:8:20}_nipsrb_udp_${AT}.log
                {
                    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${FREE_GPUS[*]}") nohup python main.py $CFG --use_generated_oladata true --attn_type $AT >> $LOG_FILE 2>&1
                } &
                training_pid=$!
                for used_gpu in "${FREE_GPUS[@]}"; do
                    for ((idx=0; idx<${#GPU_LIST[@]}; idx++)); do
                        if [ "${GPU_LIST[$idx]}" == "$used_gpu" ]; then
                            PID_LIST[$idx]=$training_pid
                        fi
                    done
                done
                echo CUDA_DEVICES_$(IFS=,; echo "${FREE_GPUS[*]}")_CFG_${CFG:8:20}_PID:${training_pid}_LOG:$LOG_FILE
                sleep 2s
                break
            fi
        done
    done
done
wait

echo --------------------- finish -------------------
date
