import os
import clip
import torch


## PAOT
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# text_inputs = clip.tokenize('A photo of left Atrium').to(device)
text_inputs = clip.tokenize('An image containing the left Atrium, with the rest being background').to(device)
print(text_inputs.shape,text_inputs.dtype)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding.pth')

