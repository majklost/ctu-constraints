# Makefile for generating artificial datasets.
# example usage:
# make rigid                               # large rigid only
# make deformed                            # deformation only, no rigid pre-transform
# make rigid_deformed                      # large rigid + deformation
# make all                                 # all dataset variants
# make rigid TRAIN_N=20000 TRAIN_SEED=7    # override on the fly
# make clean

# ---- config ----
PYTHON       ?= python
GEN_SCRIPT   := scripts/create_artificial_dataset.py
DATA_ROOT    := data/artificial

TRAIN_N      ?= 2000
VAL_N        ?= 100

TRAIN_SEED   ?= 42
VAL_SEED     ?= 43

RIGID_MODE           ?= large
DEFORMED_MODE        ?= none
RIGID_DEFORMED_MODE  ?= large

DATASETS := rigid deformed rigid_deformed
SPLIT_TARGETS := \
	rigid-trn rigid-val \
	deformed-trn deformed-val \
	rigid_deformed-trn rigid_deformed-val
KEEP_DIRS := downloaded custom

# ---- helper: generate one split for one dataset variant ----
# usage: $(call gen_split,dataset_name,generator_type,rigid_mode,split_name,num_samples,seed)
define gen_split
$(PYTHON) $(GEN_SCRIPT) $(5) \
	--output_dir $(DATA_ROOT)/$(1)/$(4) \
	--seed $(6) \
	--generator_type $(2) \
	--rigid_mode $(3)
endef

# ---- dataset targets ----
.PHONY: $(DATASETS) $(SPLIT_TARGETS) all clean

rigid: rigid-trn rigid-val

rigid-trn:
	$(call gen_split,rigid,rigid,$(RIGID_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

rigid-val:
	$(call gen_split,rigid,rigid,$(RIGID_MODE),val,$(VAL_N),$(VAL_SEED))

deformed: deformed-trn deformed-val

deformed-trn:
	$(call gen_split,deformed,deformed,$(DEFORMED_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

deformed-val:
	$(call gen_split,deformed,deformed,$(DEFORMED_MODE),val,$(VAL_N),$(VAL_SEED))

rigid_deformed: rigid_deformed-trn rigid_deformed-val

rigid_deformed-trn:
	$(call gen_split,rigid_deformed,deformed,$(RIGID_DEFORMED_MODE),trn,$(TRAIN_N),$(TRAIN_SEED))

rigid_deformed-val:
	$(call gen_split,rigid_deformed,deformed,$(RIGID_DEFORMED_MODE),val,$(VAL_N),$(VAL_SEED))

all: $(SPLIT_TARGETS)

clean:
	@echo "Cleaning generated datasets (keeping $(KEEP_DIRS))..."
	find $(DATA_ROOT) -mindepth 1 -maxdepth 1 \
		$(foreach d,$(KEEP_DIRS),! -name $(d)) \
		-exec rm -rf {} +
