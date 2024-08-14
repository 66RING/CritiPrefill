# run:
# 	python main.py -m $(MODELS_DIR)/tinyllama-110M
# 	# python main.py -m $(MODELS_DIR)/LLaMA-2-1.1B-2bit-groupsize32
# 	# python main_api.py -m $(MODELS_DIR)/LLaMA-2-1.1B-2bit-groupsize32

run naive:
	python main_api.py -m $(MODELS_DIR)/tinyllama-110M

e eattn:
	python main_api.py -m $(MODELS_DIR)/tinyllama-110M -e

l longbench:
	bash ./run_pred.sh

n needle:
	bash ./run_needle.sh

