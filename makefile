run naive:
	python main.py -m $(MODELS_DIR)/tinyllama-110M

e eattn:
	python main.py -m $(MODELS_DIR)/tinyllama-110M -e --prefill_only --segment_size 1024 --threshold_len 1024 --budgets 512

