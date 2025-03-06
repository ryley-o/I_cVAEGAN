# I_cVAEGAN

# Conditional Variational Autoencoder Generative Adversarial Network Hybrid(cVAE-GAN)

A small, efficient conditional cVAE-GAN implementation for 64x64 images using TensorFlow/Keras. This implementation is designed to be easily quantizable and deployable to blockchain via TensorFlow.js.

> _Note: LLM agent-based coding was used heavily when creating the code for this model. AI was embraced for reasons closely related to why this model was created._

## Features

- [x] Conditional Variational Autoencoder (cVAE)
- [x] Generative Adversarial Network (GAN)
- [x] Conditional consistency and contrastive regularization
- [x] Training schedule to blend cVAE's and GAN's strengths while focusing in interpolation stability
- [x] Hyper-compact model
- [x] Quantizable
- [x] Deployable to blockchain

## Usage

```python
# Train the model
python3 train_vaegan3_revb.py

# Export TFJS quantized model
python3 export_vaegan3_revb_tfjs.py

# Run python server
python3 -m http.server 8000

# Run the model in the browser, refresh for new images
open http://localhost:8000/simple_vaegan3_interp.html

# if you want to generate a grid of output images, run the following command
python3 generate_grid_vaegan3.py
```
