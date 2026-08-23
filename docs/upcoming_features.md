# Upcoming features checklist

Checklist of features to be implemented in this repo

# 0. Revision

- [ ] revise if into metrics I pass probabilities or raw logits, maybe unify or offer clear contract
- [ ] Decide how to stabilize deformable registration with velocity-magnitude
      and spatial-smoothness regularization: implement them as diagnostic metrics and loss terms. Include displacement magnitude, warped foreground mass,
      and off-grid sampling diagnostics when evaluating the choice.
- [ ] replace name affine with name rigid across the codebase (affine was wrongly chosen)

# 1. Model weight saving

- [ ] verify if saving torch.save is enough if we suppose that training wont be resumed
- [ ] think about syncing strategy, if need to sync locally for some testing (or how to know which files exist)

# 2.Harder dataset creation

# 3. Multiple template deformation

# 4. Connection of external dataset

# 5. Branch architecture

# 6. Anatomically constrained neural network implementation and TEDS-NET implementation
