"""
Here we use affine prealignment and later predict deformation field

We use them in decoupled manner - we pass GT into projector
similar to ex3 but more losses and better metrics measurement

Tested pairs (segm + reg):
- BCE + OneSideSDFSquared x
- BCE + OneSideSDFPlain x
- BCE + BCE x
- BCE + CentroidLoss x
- BCE + BlurredLoss x
- BCE + DSDF_MSE x
- BCE + SDFTEMPLATE_MSE x
- BCE + SDFTEMPLATE_OneSideSDFSQUARE x
- OneSideSDFSquared + OneSideSDFSquared x
- OneSideSDFPlain + OneSideSDFPlain x
- UNET
"""
