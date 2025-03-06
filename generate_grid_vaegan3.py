import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import argparse

# Configuration defaults
DEFAULT_MODEL_PATH = './models_revb/vaegan3_revb_final'
DEFAULT_OUTPUT_DIR = './grid_samples_vaegan3'
DEFAULT_GRID_SIZE = 10  # 10x10 grid = 100 images
DEFAULT_LATENT_DIM = 64  # Match the latent dimension in the model
DEFAULT_CONDITION_DIM = 15
DEFAULT_BATCH_SIZE = 32  # Updated to match the model's expected batch size of 32

def get_random_latent_vector(batch_size, latent_dim):
    """Generate random latent vectors for the VAE-GAN model."""
    return tf.random.normal(shape=(batch_size, latent_dim))

def get_random_conditions(batch_size, condition_dim, gender_value=2.0):
    """Generate random condition vectors with one-hot encoding for some attributes."""
    conditions = np.zeros((batch_size, condition_dim), dtype=np.float32)
    
    # Select random emotions (first 10 dimensions)
    for i in range(batch_size):
        emotion_idx = np.random.randint(0, 10)
        emotion_idx2 = np.random.randint(0, 10)
        blend_amount = np.random.rand()
        # blend_amount = 1
        if emotion_idx == emotion_idx2:
            blend_amount = 1.0
        conditions[i, emotion_idx] = blend_amount
        if emotion_idx != emotion_idx2:
            conditions[i, emotion_idx2] = 1.0 - blend_amount
        
        # Set class (dimensions 10-14)
        class_idx = np.random.randint(10, 14)
        class_idx2 = np.random.randint(10, 14)
        blend_amount_class = np.random.rand()
        # blend_amount_class = 1
        if class_idx == class_idx2:
            blend_amount_class = 1.0
        conditions[i, class_idx] = blend_amount_class
        if class_idx != class_idx2:
            conditions[i, class_idx2] = 1.0 - blend_amount_class
        
        # Set gender (last dimension)
        # gender_idx = np.random.choice([13, 14])
        gender_idx = 14
        conditions[i, gender_idx] = 1
        print(f"index: {i}")
        print(f"conditions: {conditions[i]}")

    
    
    return tf.convert_to_tensor(conditions, dtype=tf.float32)

def generate_grid_image(model_path, output_dir, grid_size, latent_dim, condition_dim, batch_size, random_seed=None, gender_value=2.0):
    """Generate a grid of images from the model using random latent vectors."""
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    else:
        # Use current time for randomness
        seed = int(time.time())
        np.random.seed(seed)
        tf.random.set_seed(seed % 10000)
        print(f"Using random seed: {seed}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    # Load the saved model using tf.saved_model.load
    model = tf.saved_model.load(model_path)
    
    # Check available signatures
    print("Available signatures:", list(model.signatures.keys()))
    
    # Get the serving default signature
    serving_fn = model.signatures["serving_default"]
    
    # Print input and output specs
    print("\nModel Signature Details:")
    print("\nSignature: serving_default")
    print("  Inputs:")
    for input_name, input_tensor in serving_fn.structured_input_signature[1].items():
        print(f"    {input_name}: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
    
    print("  Outputs:")
    for output_name, output_tensor in serving_fn.structured_outputs.items():
        print(f"    {output_name}: shape={output_tensor.shape}, dtype={output_tensor.dtype}")
    
    total_images = grid_size * grid_size
    print(f"\nGenerating {total_images} images...")
    
    # Generate images in batches to match the model's expected batch size
    num_batches = (total_images + batch_size - 1) // batch_size
    generated_images = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        curr_batch_size = end_idx - start_idx
        
        # If the current batch size is smaller than the model's expected batch size,
        # we need to generate a full batch and then take only what we need
        actual_batch_size = batch_size if curr_batch_size == batch_size else batch_size
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (images {start_idx+1}-{end_idx})")
        
        # Generate random conditions for this batch
        conditions = get_random_conditions(actual_batch_size, condition_dim, gender_value)
        
        # Call the model's serving function
        output = serving_fn(
            images=tf.ones((actual_batch_size, 64, 64, 3), dtype=tf.float32),  # Dummy images
            conditions=conditions
        )
        
        # Get the generated images from the output
        batch_images = output["generated_images"].numpy()
        
        # If this is the last batch and we need only a subset of images
        if curr_batch_size < batch_size:
            batch_images = batch_images[:curr_batch_size]
            
        generated_images.append(batch_images)
    
    # Concatenate all batches
    generated_images = np.concatenate(generated_images, axis=0)
    
    # Only keep the images we need for the grid
    generated_images = generated_images[:total_images]
    
    # Create the figure for the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2*grid_size, 2*grid_size))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # Handle the case where grid_size is 1 (matplotlib returns a single axes, not a 2D array)
    if grid_size == 1:
        axes = np.array([[axes]])
    
    # Plot each image in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j]
            ax.imshow(np.clip(generated_images[idx], 0, 1))  # Clip values to [0, 1] range
            ax.axis('off')
    
    # Save the grid
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f'vaegan3_grid_{grid_size}x{grid_size}_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Grid image saved to {output_path}")
    return output_path

def main():
    """Main function to parse arguments and generate the grid image."""
    parser = argparse.ArgumentParser(description='Generate a grid of images from the VAE-GAN3 model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to the model directory (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for generated images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--grid-size', type=int, default=DEFAULT_GRID_SIZE,
                        help=f'Size of the grid (NxN) (default: {DEFAULT_GRID_SIZE})')
    parser.add_argument('--latent-dim', type=int, default=DEFAULT_LATENT_DIM,
                        help=f'Dimension of the latent space (default: {DEFAULT_LATENT_DIM})')
    parser.add_argument('--condition-dim', type=int, default=DEFAULT_CONDITION_DIM,
                        help=f'Dimension of the condition vector (default: {DEFAULT_CONDITION_DIM})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for model inference (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses time)')
    parser.add_argument('--gender-value', type=float, default=2.0,
                        help='Value to use for gender attribute (default: 2.0)')
    
    args = parser.parse_args()
    
    try:
        output_path = generate_grid_image(
            model_path=args.model,
            output_dir=args.output,
            grid_size=args.grid_size,
            latent_dim=args.latent_dim,
            condition_dim=args.condition_dim,
            batch_size=args.batch_size,
            random_seed=args.seed,
            gender_value=args.gender_value
        )
        print(f"Successfully generated grid image at: {output_path}")
    except Exception as e:
        print(f"Error generating grid image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 