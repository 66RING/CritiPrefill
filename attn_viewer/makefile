r run:
	# @CUDA_VISIBLE_DEVICES=3 python main.py -m $(MODELS_DIR)/tinyllama-110M
	python main.py -m $(MODELS_DIR)/tinyllama-110M
	# python main.py -m $(MODELS_DIR)/LWM-Text-Chat-1M
	# @CUDA_VISIBLE_DEVICES=3 python main.py -m $(MODELS_DIR)/Llama-3-8B-Instruct-Gradient-1048k

m mrun:
	python main_multithread.py -m $(MODELS_DIR)/tinyllama-110M

