# Makefile for generating artificial datasets.
# example usage:
# make affine                              # large affine only
# make deformed                            # deformation only, no affine pre-transform
# make affine_deformed                     # large affine + deformation
# make all                                 # all dataset variants
# make affine TRAIN_N=20000 TRAIN_SEED=7   # override on the fly
# make clean

# ---- config ----
PYTHON       ?= python
GEN_SCRIPT   := scripts/create_artificial_dataset.py
DATA_ROOT    := data/artificial

TRAIN_N      ?= 2000
VAL_N        ?= 100

TRAIN_SEED   ?= 42
VAL_SEED     ?= 43

AFFINE_MODE          ?= large
DEFORMED_MODE        ?= none
AFFINE_DEFORMED_MODE ?= large

DATASETS := affine deformed affine_deformed
SPLIT_TARGETS := \
	affine-trn affine-val \
	deformed-trn deformed-val \
	affine_deformed-trn affine_deformed-val
KEEP_DIRS := downloaded custom

# ---- helper: generate one split for one dataset variant ----
# usage: $(call gen_split,dataset_name,generator_type,affine_mode,split_name,num_samples,seed)
define gen_split
$(PYTHON) $(GEN_SCRIPT) $(5) \
	--output_dir $(DATA_ROOT)/$(1)/$(4) \
	--seed $(6) \
	--generator_type $(2) \
	--affine_mode $(3)
endef

# ---- dataset targets ----
.PHONY: $(DATASETS) $(SPLIT_TARGETS) all clean

affine: affine-trn affine-val

affine-trn:
	$(call gen_split,affine,affine,$(AFFINE_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

affine-val:
	$(call gen_split,affine,affine,$(AFFINE_MODE),val,$(VAL_N),$(VAL_SEED))

deformed: deformed-trn deformed-val

deformed-trn:
	$(call gen_split,deformed,deformed,$(DEFORMED_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

deformed-val:
	$(call gen_split,deformed,deformed,$(DEFORMED_MODE),val,$(VAL_N),$(VAL_SEED))

affine_deformed: affine_deformed-trn affine_deformed-val

affine_deformed-trn:
	$(call gen_split,affine_deformed,deformed,$(AFFINE_DEFORMED_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

affine_deformed-val:
	$(call gen_split,affine_deformed,deformed,$(AFFINE_DEFORMED_MODE),val,$(VAL_N),$(VAL_SEED))

all: $(SPLIT_TARGETS)

clean:
	@echo "Cleaning generated datasets (keeping $(KEEP_DIRS))..."
	find $(DATA_ROOT) -mindepth 1 -maxdepth 1 \
		$(foreach d,$(KEEP_DIRS),! -name $(d)) \
		-exec rm -rf {} +
