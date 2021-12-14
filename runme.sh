make
./layer_norm.out 10 10000000 | grep '^LayerNorm'
./layer_norm.out 100 1000000 | grep '^LayerNorm'
./layer_norm.out 1000 100000 | grep '^LayerNorm'
./layer_norm.out 10000 10000 | grep '^LayerNorm'
./layer_norm.out 100000 1000 | grep '^LayerNorm'
./layer_norm.out 1000000 100 | grep '^LayerNorm'
./layer_norm.out 10000000 10 | grep '^LayerNorm'
