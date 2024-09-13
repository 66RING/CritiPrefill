run naive:
	python main.py -m $(MODELS_DIR)/tinyllama-110M

e eattn:
	python main.py -m $(MODELS_DIR)/tinyllama-110M -e --segment_size 1024 --threshold_len 1024 --budgets 512 --layer_fusion

i install:
	pip install -e . && pip install flash_attn==2.5.8 --no-build-isolation

