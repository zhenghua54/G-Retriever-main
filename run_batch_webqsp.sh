
for seed in 0 1 2 3
do
python train.py --dataset webqsp --model_name graph_llm --llm_model_name 7b --project Hd239_webqsp_GATv2_retriever_7b --seed $seed

done
