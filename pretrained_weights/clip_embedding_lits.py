import os
import clip
import torch


## PAOT


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# text_inputs = clip.tokenize(' A photo of a kidney and kidney tumor').to(device)

text_inputs = clip.tokenize('An image containing the liver, and liver tumor, with the rest being background').to(device)

print(text_inputs.shape,text_inputs.dtype)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding_lits.pth')

