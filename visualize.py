import torch
import torchvision
from scipy.stats import truncnorm

def random(model,out_path,tmp=0.4, n=9, truncate=False):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        dataset_size = model.embeddings.weight.size()[0]
        dim_z = model.embeddings.weight.size(1)
        if truncate:
            embeddings = truncnorm(-tmp, tmp).rvs(n * dim_z).astype("float32").reshape(n, dim_z)
        else:
            embeddings = np.random.normal(0, tmp, size=(n, dim_z)).astype("float32")
        embeddings = torch.tensor(embeddings,device=device)
        batch_size = embeddings.size()[0]
        image_tensors = model(embeddings)
        torchvision.utils.save_image(
                image_tensors,
                out_path,
                nrow=int(batch_size ** 0.5),
                normalize=True,
            )