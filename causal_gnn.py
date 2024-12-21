from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import bnlearn as bn

data = bn.import_example('asia')
X = data

# default parameterss
G = grasp(X)
print(G)

# Visualization using pydot
pyd = GraphUtils.to_pydot(G)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()