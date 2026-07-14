# Makefile for generating artificial datasets using different generator types.
# example usage:
# make affine              # generates data/artificial/affine/{train,val,test}
# make deformed
# make all                 # all generator types
# make affine TRAIN_N=20000 TRAIN_SEED=7   # override on the fly
# make clean

# ---- config ----
PYTHON       ?= python
GEN_SCRIPT   := scripts/create_artificial_dataset.py
DATA_ROOT    := data/artificial

TRAIN_N      ?= 2000
VAL_N        ?= 100
TST_N 	  ?= 100

TRAIN_SEED   ?= 42
VAL_SEED     ?= 43
TST_SEED     ?= 44
AFFINE_MODE ?= small

GEN_TYPES    := affine deformed both
KEEP_DIRS := downloaded custom

# ---- helper: generate one split for one generator type ----
# usage: $(call gen_split,generator_type,split_name,num_samples,seed)
define gen_split
$(PYTHON) $(GEN_SCRIPT) $(3) \
	--output_dir $(DATA_ROOT)/$(1)/$(2) \
	--seed $(4) \
	--generator_type $(1) \
	--affine_mode $(AFFINE_MODE)
endef

# ---- per-generator-type targets ----
.PHONY: $(GEN_TYPES) all clean

$(GEN_TYPES):
	$(call gen_split,$@,trn,$(TRAIN_N),$(TRAIN_SEED))
	$(call gen_split,$@,val,$(VAL_N),$(VAL_SEED))
	$(call gen_split,$@,tst,$(TST_N),$(TST_SEED))
all: $(GEN_TYPES)

clean:
	@echo "Cleaning generated datasets (keeping $(KEEP_DIRS))..."
	find $(DATA_ROOT) -mindepth 1 -maxdepth 1 \
		$(foreach d,$(KEEP_DIRS),! -name $(d)) \
		-exec rm -rf {} +
