# normal training: [r, g, b] color prediction for each vertex
# python3 Test/gnn.py --config Configs/gnn_kraken.json --configD Configs/discriminator.json --model_idx 0 \
# --model_path /home/kcdharma/results/arete-realsim/gnn_v.pth --type vertex

# dual training: [r, g, b] color prediction for each face
# gnn_d1: gnn model trained on dual graph with output texture resolution: 1 [1 x 1 x 3]
python3 Test/gnn.py --config Configs/gnn_kraken.json --configD Configs/discriminator.json \
--model_path /home/kcdharma/results/arete-realsim/gnn_d1.pth --type dual --out_res 1 --model_idx 0

# dual training: [res x res x 3] texture map per face
# gnn_d2: gnn model trained on dual graph with output texture resolution: 2 [2 x 2 x 3]
# python3 Test/gnn.py --config Configs/gnn_kraken.json --configD Configs/discriminator.json --model_idx 0 \
# --model_path /home/kcdharma/results/arete-realsim/gnn_d2.pth --type dual --out_res 2