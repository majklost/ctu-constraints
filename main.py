from constraints.test import f
from pathlib import Path

print(Path(__file__).name)

# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# %%
t = np.random.random((200, 200))

# %%
plt.imshow(t)
# %%
p = pd.DataFrame({"prd": [1, 2, 3, 4]})
p.head()
# %%
import torch

torch.cuda.is_available()

# %%
