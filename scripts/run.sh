CUDA_VISIBLE_DEVICES=2 python main.py \
	configs/mae3d.yaml \ 
	--run_name='test'

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/vit4d.yaml \ 
	--run_name='test'