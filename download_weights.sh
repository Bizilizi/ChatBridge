mkdir -p vicuna_weights/13b/
huggingface-cli download Vision-CAIR/vicuna --local-dir vicuna_weights/13b/ --local-dir-use-symlinks False

mkdir beats_weights
cd beats_weights
wget https://huggingface.co/nsivaku/nithin_checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt?download=true -O BEATs_iter3_plus_AS2M.pt







