TASK=$1
ALGO=$2

echo "Experiments started."
echo $(seq 30 40) | tr ' ' '\n' | parallel  -I% --colsep ' ' --max-args 1 -j 3 "echo % start&python ${ALGO}.py --env-id $TASK --seed % --wandb-entity quangr  --track > ${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_%.txt 2>&1"
echo "Experiments ended."

