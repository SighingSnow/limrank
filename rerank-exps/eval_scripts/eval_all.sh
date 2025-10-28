#!/bin/bash
model_name=$1
model="models/${model_name}"

echo "Model: $model"

subjects=(
earth_science
biology
economics 
psychology
robotics
stackoverflow
sustainable_living
leetcode
pony
aops
theoremqa_questions
theoremqa_theorems
)

for subject in "${subjects[@]}"; do
    bash eval_scripts/launch_job.sh "$model" BrightRetrieval "$subject" 2
done

# Evaluate FollowIR
datasets=(
Robust04
Core17
News21
)
for dataset in "${datasets[@]}"; do
    bash eval_scripts/launch_job.sh "$model" "$dataset" default 2
done

datasets=(
LitSearchRetrieval
)
for dataset in "${datasets[@]}"; do
    bash eval_scripts/launch_job.sh "$model" "$dataset" default 2
done