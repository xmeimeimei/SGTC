import os
import clip
import torch


## PAOT


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# text_inputs = clip.tokenize('A photo of a kidney and kidney tumor').to(device)  #  v1
# text_inputs = clip.tokenize('A computerized tomography of a kidney and kidney tumor ').to(device)  # v2
# text_inputs = clip.tokenize('An image containing the left kidney, right kidney, and kidney tumor, with the rest being background').to(device)  # v4
# text_inputs = clip.tokenize('There is a kidney and kidney tumor in this photo').to(device)  # v4
text_inputs = clip.tokenize('There is a kidney and kidney tumor in this computerized tomography').to(device)  # v5

print(text_inputs.shape,text_inputs.dtype)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding_kits_v5.pth')

