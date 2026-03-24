
file_path=/text2sql/verl-tool/data/QAdataset
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/text2sql/verl-tool/base_model/e5-base-v2

port=$(shuf -i 30000-31000 -n 1)

python3 verl_tool/servers/tools/utils/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path\
     &
retriever_pid=$!


python3 -m verl_tool.servers.serve \
    --host 0.0.0.0 \
    --port $port \
    --tool_type "search_retrieval" \
    --workers_per_tool 8 \