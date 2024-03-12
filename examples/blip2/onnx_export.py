# Code comments added by the software engineer:

import os # Importing the os module for various operating system dependent functionalities

import requests # Importing the requests module for making HTTP requests
import torch # Importing the PyTorch library for deep learning
from PIL import Image # Importing the Image module from the Pillow library for image processing

# In docker environment
if os.getcwd().startswith('/workspace'): # Checking if the current working directory is within /workspace in a Docker environment
    os.environ['TORCH_HOME'] = '/workspace/.cache' # Setting the TORCH_HOME environment variable to /workspace/.cache in a Docker environment
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache' # Setting the TRANSFORMERS_CACHE environment variable to /workspace/.cache in a Docker environment

from lavis.models import load_model_and_preprocess # Importing the load_model_and_preprocess function from the lavis.models module

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' # Defining the URL of the image to be processed
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB') # Downloading the image from the URL and converting it to RGB format

device = torch.device("cuda") if torch.cuda.is_available() else "cpu" # Determining the device (GPU or CPU) to be used for processing

model, vis_processors, _ = load_model_and_preprocess( # Loading the pre-trained model and preprocessors
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) # Preprocessing the image and moving it to the specified device

torch.save(model.query_tokens, 'query_tokens.pt') # Saving the query tokens to a file named query_tokens.pt

if not os.path.exists('image.pt'): # Checking if a file named image.pt already exists
    torch.save(image, 'image.pt') # Saving the preprocessed image to a file named image.pt

txt_caption = model.generate({ # Generating a text caption for the image
    "image": image,
    "prompt": "Question: which city is this? Answer:"
})
print(txt_caption)

visual_wrapper = torch.nn.Sequential(model.visual_encoder, model.ln_vision) # Defining a sequential model using the visual encoder and layer normalization for the vision module
visual_wrapper.float() # Casting the model to float

image_embeds = visual_wrapper(image) # Generating the image embeddings using the visual wrapper

os.system('mkdir -p ./onnx/visual_encoder') # Creating a directory named visual_encoder within the onnx directory

torch.onnx.export(visual_wrapper.cpu(), # Exporting the visual wrapper model to ONNX format
                  image.cpu(),
                  './onnx/visual_encoder/visual_encoder.onnx',
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {
                      0: 'batch'
                  }})

image_atts = torch.ones(image_embeds.size()[:-1], # Defining image attributes as a tensor of ones with the same shape as the image embeddings (excluding the last dimension)
                        dtype=torch.long).to(image.device)
query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1) # Expanding the query tokens to match the shape of the image embeddings

class Qformer_wrapper(torch.nn.Module): # Defining a class named Qformer_wrapper that inherits from the torch.nn.Module class

    def __init__(self, Qformer, opt_proj): # Initializing the Qformer_wrapper class with two parameters: Qformer and opt_proj
        super().__init__() # Calling the constructor of the parent class (torch.nn.Module)
        self.model = Qformer # Assigning the Qformer parameter to the model attribute
        self.opt_proj = opt_proj # Assigning the opt_proj parameter to the opt_proj attribute

    def forward(self, query_tokens, image_embeds, image_atts): # Defining the forward method for the Qformer_wrapper class
        query_output = self.model(query_embeds=query_tokens, # Passing the query_tokens, image_embeds, and image_atts through the model
                                  encoder_hidden_states=image_embeds,
                                  encoder_attention_mask=image_atts,
                                  return_dict=True)
        return self.opt_proj(query_output.last_hidden_state) # Returning the output of the opt_proj applied to the last hidden state of the query_output

q_wrapper = Qformer_wrapper
