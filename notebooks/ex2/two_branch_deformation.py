# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ctu-constraints
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Two branch architecture
- with images generated as deformed
"""

# %%
# %load_ext autoreload
# %autoreload 2
# from constraints.models.deform_only import TwoBranch
# from constraints.generators import ArteryGeneratorDeformed
# from constraints.visu import show_torch_image
# from constraints import REPO_ROOT
import pytorch_lightning as pl
import neurite as ne
import torch

# %%
print(REPO_ROOT)

# %%
dataset_trn = ArteryGeneratorDeformed(num_samples=1000, fixed_seed=43, magnitude=7.0, integrations=2, scales=14,fractal_mode="blur")
dataset_val = ArteryGeneratorDeformed(num_samples=100, fixed_seed=43, magnitude=4.0, integrations=3, scales=15, fractal_mode="blur")
for i in range(5):
    trn1 = dataset_trn[i]
    show_torch_image(trn1["img"], cmap="gray")
    if i ==0:
        print(trn1.keys())
    # for k in trn1.keys():
        
    #     print(k, trn1[k].shape)
    #     if k == "field": continue
    #     show_torch_image(trn1[k],  cmap="gray" if k == "img" else None)


# %%
class LitTwoBranch(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TwoBranch(
            source_channels=1,
            target_channels=3,
            nb_features=[32, 32, 32, 32],
            integration_steps=5,
        )
        
        self.template_loss_fn =torch.nn.MSELoss() #TODO centroid or SDF loss
        
        self.grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
        self.grad_loss_weight = 0.05
        self.segmentation_loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, img: torch.Tensor, template: torch.Tensor|None = None):
        return self.model(img, template)
    
    def _shared_step(self,batch,stage):
        template = batch["template"]
        img = batch["img"]
        mask = batch["mask"]
        segmentation_logits,field, warped_template = self(img, template)
        template_loss = self.template_loss_fn(warped_template, mask)
        grad_loss = self.grad_loss_fn(field)
        segmentation_loss = self.segmentation_loss_fn(segmentation_logits, mask)
        # keep template_loss now for debugging
        loss = template_loss + self.grad_loss_weight * grad_loss + segmentation_loss
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_template_loss", template_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_grad_loss", grad_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_segmentation_loss", segmentation_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss        
        
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        template = batch["template"]
        mask = batch["mask"]
        img = batch["img"]
        segmentation_logits,field, warped_template = self(img, template)
        return {
            "segmentation_logits": segmentation_logits,
            "field": field,
            "warped_template": warped_template,
            "mask": mask,
            "img": img
        }
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return optimizer

# %%
EPOCHS= 100
BATCH_SIZE = 4
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path

logger = CSVLogger('logs', name='two_branch_deformation')
model = LitTwoBranch()

# overfit one batch
trn_loader = torch.utils.data.DataLoader(dataset_trn, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices='auto',
    logger=logger,
    overfit_batches=1,
)
trainer.fit(model, trn_loader, val_loader)

# %%
#save the overfitted model predictions
batch = next(iter(trn_loader))
res = model.predict_step(batch,0)


# %%
print(str(Path(".").resolve()))
print(list(Path(".").iterdir()))

# %%
print(res.keys())
output_dir = Path("./notebooks/ex2/overfit_output")
output_dir.mkdir(parents=True, exist_ok=True)
warped_templates = res["warped_template"]
masks = res["mask"]
print(warped_templates.shape, masks.shape)
for i in range(warped_templates.shape[0]):
    show_torch_image(warped_templates[i],save_path=output_dir / f"warped_template_{i}.png")
    show_torch_image(masks[i],save_path=output_dir / f"mask_{i}.png")

# %%
