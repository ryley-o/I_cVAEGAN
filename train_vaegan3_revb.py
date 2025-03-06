import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import time
import datetime
from vaegan3_revb import get_vaegan3_model, VAEGAN3

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 32
EPOCHS = 3600
IMAGE_SIZE = 64
LATENT_DIM = 1
CONDITION_DIM = 15
CONSISTENCY_WEIGHT = 3.0 
CONTRASTIVE_WEIGHT = 1.0 
TEMPERATURE = 0.5 
CHECKPOINT_DIR = './checkpoints_vaegan3_revb'
MODEL_SAVE_DIR = './models_revb'
SAMPLE_DIR = './samples_vaegan3_revb'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def load_attributes(attr_path):
    """Load attributes from file."""
    attributes = {}
    with open(attr_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Parse line
            parts = line.strip().split()
            if len(parts) > 1:
                filename = parts[0]
                attrs = [float(attr) for attr in parts[1:]]
                attributes[filename] = attrs
    return attributes

def load_and_preprocess_image(img_path):
    """Load and preprocess an image."""
    img = Image.open(img_path)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize if needed
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    return img_array.astype(np.float32)

def prepare_dataset(img_dir, attr_path):
    """Prepare dataset from images and attributes."""
    print(f"Loading attributes from {attr_path}")
    attributes = load_attributes(attr_path)
    
    print(f"Loading images from {img_dir}")
    images = []
    conditions = []
    
    # Get all image files
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            if img_file in attributes:
                img_path = os.path.join(img_dir, img_file)
                try:
                    img_array = load_and_preprocess_image(img_path)
                    images.append(img_array)
                    conditions.append(attributes[img_file])
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            else:
                print(f"Warning: No attributes found for {img_file}")
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    conditions = np.array(conditions, dtype=np.float32)
    
    print(f"Dataset prepared: {len(images)} images with shape {images.shape}")
    print(f"Conditions shape: {conditions.shape}")
    
    return images, conditions

def generate_and_save_samples(model, epoch, conditions, save_dir):
    """Generate and save sample images using the model."""
    # Generate images using random latent vectors
    batch_size = len(conditions)
    z = tf.random.normal(shape=(batch_size, model.latent_dim))
    generated_images = model.generator([z, conditions])
    
    # Convert to numpy and adjust range [0, 1] to [0, 255]
    images = generated_images.numpy()
    
    # Create grid of images
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        title = ', '.join([f"{c:.1f}" for c in conditions[i]])
        plt.title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
    plt.close()
    
    # Save individual images
    for i, img in enumerate(images[:min(16, len(images))]):
        img_array = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array)
        img_pil.save(os.path.join(save_dir, f'sample_{epoch}_{i}.png'))

def generate_reconstructions(model, images, conditions, epoch, save_dir):
    """Generate and save reconstructions of input images."""
    # Get encoder output
    z_mean, z_log_var = model.encoder([images, conditions])
    
    # Sample from latent space
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, model.latent_dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    # Generate reconstructed images
    reconstructed_images = model.generator([z, conditions])
    
    # Plot original vs reconstructed
    plt.figure(figsize=(12, 6))
    
    for i in range(min(8, len(images))):
        # Original image
        plt.subplot(2, 8, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        if i == 0:
            plt.title('Original', fontsize=10)
        
        # Reconstructed image
        plt.subplot(2, 8, i+9)
        plt.imshow(reconstructed_images[i])
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstructions_epoch_{epoch}.png'))
    plt.close()

def generate_interpolations(model, epoch, save_dir):
    """Generate and save interpolations in the latent and condition spaces."""
    # Create a grid of images showing interpolation in latent space
    # and interpolation in condition space
    
    # Fixed random seed for comparison across epochs
    np.random.seed(42)
    
    # 1. Latent space interpolation with fixed condition
    n_samples = 10
    condition = np.zeros((1, CONDITION_DIM), dtype=np.float32)
    condition[0, 0] = 1.0  # Use first condition as fixed
    
    z_1 = tf.random.normal(shape=(1, LATENT_DIM), seed=42)
    z_2 = tf.random.normal(shape=(1, LATENT_DIM), seed=43)
    
    alphas = np.linspace(0, 1, n_samples)
    z_interp = np.zeros((n_samples, LATENT_DIM), dtype=np.float32)
    
    for i, alpha in enumerate(alphas):
        z_interp[i] = (1 - alpha) * z_1.numpy() + alpha * z_2.numpy()
    
    z_interp = tf.convert_to_tensor(z_interp, dtype=tf.float32)
    conditions_repeated = tf.repeat(condition, repeats=n_samples, axis=0)
    
    latent_interp_images = model.generator([z_interp, conditions_repeated])
    
    # 2. Condition space interpolation with fixed latent vector
    z = tf.random.normal(shape=(1, LATENT_DIM), seed=42)
    
    # Interpolate between two different conditions
    condition_1 = np.zeros((1, CONDITION_DIM), dtype=np.float32)
    condition_2 = np.zeros((1, CONDITION_DIM), dtype=np.float32)
    
    # Set two different conditions
    condition_1[0, 0] = 1.0  # First condition
    condition_2[0, 1] = 1.0  # Second condition
    
    condition_interp = np.zeros((n_samples, CONDITION_DIM), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        condition_interp[i] = (1 - alpha) * condition_1 + alpha * condition_2
    
    condition_interp = tf.convert_to_tensor(condition_interp, dtype=tf.float32)
    z_repeated = tf.repeat(z, repeats=n_samples, axis=0)
    
    condition_interp_images = model.generator([z_repeated, condition_interp])
    
    # Create a figure showing the interpolations
    plt.figure(figsize=(15, 6))
    
    # Plot latent space interpolation
    for i in range(n_samples):
        plt.subplot(2, n_samples, i+1)
        plt.imshow(latent_interp_images[i])
        plt.title(f"α={alphas[i]:.1f}", fontsize=8)
        plt.axis('off')
        
        if i == 0:
            plt.ylabel('Latent Space\nInterpolation', fontsize=10)
    
    # Plot condition space interpolation
    for i in range(n_samples):
        plt.subplot(2, n_samples, n_samples+i+1)
        plt.imshow(condition_interp_images[i])
        plt.axis('off')
        
        if i == 0:
            plt.ylabel('Condition Space\nInterpolation', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'interpolations_epoch_{epoch}.png'))
    plt.close()

def visualize_condition_consistency(model, epoch, save_dir):
    """Visualize how well the model preserves conditions during image generation."""
    # Generate images with various conditions and check if the condition predictor
    # can recover the original conditions
    
    # Create a set of test conditions (one-hot encoded for clarity)
    test_conditions = np.eye(min(10, CONDITION_DIM), CONDITION_DIM, dtype=np.float32)
    conditions_tensor = tf.convert_to_tensor(test_conditions, dtype=tf.float32)
    
    # Generate images
    z = tf.random.normal(shape=(len(test_conditions), LATENT_DIM))
    generated_images = model.generator([z, conditions_tensor])
    
    # Use the condition predictor to recover conditions
    predicted_conditions = model.condition_predictor(generated_images)
    
    # Plot the original vs. predicted conditions
    plt.figure(figsize=(12, 6))
    
    # First row: Generated images
    for i in range(min(10, CONDITION_DIM)):
        plt.subplot(2, min(10, CONDITION_DIM), i+1)
        plt.imshow(generated_images[i])
        plt.title(f"Cond {i}", fontsize=8)
        plt.axis('off')
    
    # Second row: Comparison of original vs predicted conditions as bar plots
    for i in range(min(10, CONDITION_DIM)):
        plt.subplot(2, min(10, CONDITION_DIM), min(10, CONDITION_DIM)+i+1)
        
        # Get the top 3 predicted conditions
        pred_values = predicted_conditions[i].numpy()
        true_values = test_conditions[i]
        
        # Plot them side by side
        bar_width = 0.35
        index = np.arange(3)  # Top 3 conditions
        top_indices = np.argsort(pred_values)[-3:][::-1]
        
        plt.bar(index, pred_values[top_indices], bar_width, label='Pred')
        plt.bar(index + bar_width, true_values[top_indices], bar_width, label='True')
        
        plt.xticks(index + bar_width/2, [f"{j}" for j in top_indices], fontsize=7)
        plt.yticks(fontsize=7)
        plt.ylim(0, 1.0)
        if i == 0:
            plt.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'condition_consistency_{epoch}.png'))
    plt.close()

def main():
    """
    Train the VAEGAN3 model using a two-phase approach:
    
    Phase 1 (epochs 0-999):
    - Only the VAE components (encoder and generator) are trained
    - Only reconstruction and KL divergence losses are used
    - Discriminator, adversarial, and consistency losses are ignored
    - This phase helps establish good latent representations before GAN training
    
    Phase 2 (epochs 1000+):
    - All components are trained (encoder, generator, discriminator, condition predictor)
    - All losses are active (VAE, adversarial, consistency, contrastive)
    - The model leverages the representations learned in Phase 1
    """
    # Load dataset
    img_dir = '_training_img_revb'
    attr_path = os.path.join(img_dir, 'attr.txt')
    
    print("Loading dataset...")
    images, conditions = prepare_dataset(img_dir, attr_path)
    print(f"Loaded {len(images)} images with {len(conditions)} conditions")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'images': images, 
        'conditions': conditions
    })
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create model with conditional consistency and contrastive loss
    model = get_vaegan3_model(
        latent_dim=LATENT_DIM,
        condition_dim=CONDITION_DIM,
        image_size=IMAGE_SIZE,
        image_channels=3,
        consistency_weight=CONSISTENCY_WEIGHT,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        temperature=TEMPERATURE
    )
    
    # Compile the model
    model.compile(
        vae_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5),
        discriminator_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.3),
        generator_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5),
        predictor_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)  # For condition predictor
    )
    
    # Create sample conditions for visualization
    sample_conditions = conditions[:16]
    sample_images = images[:16]
    
    # Training loop
    start_time = time.time()
    
    print("\n" + "="*50)
    print("Two-Phase Training Process:")
    print("Phase 1 (epochs 0-999): VAE-only training")
    print("Phase 2 (epochs 1000+): Full GAN training")
    print("="*50 + "\n")
    
    for epoch in range(EPOCHS):
        # Set the current epoch number on the model
        model.training_epoch = epoch
        
        # Determine training phase
        if epoch < 1000:
            phase = "PHASE 1: VAE-only training"
        else:
            phase = "PHASE 2: Full GAN training"
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} - {phase}")
        print(f"Setting model.training_epoch = {epoch}")
        
        # Train for one epoch
        history = model.fit(dataset, epochs=1, verbose=1)
        
        # Generate and save visualization data
        if (epoch + 1) % 300 == 0 or epoch == 0:
            generate_and_save_samples(model, epoch+1, sample_conditions, SAMPLE_DIR)
            generate_reconstructions(model, sample_images, sample_conditions, epoch+1, SAMPLE_DIR)
            generate_interpolations(model, epoch+1, SAMPLE_DIR)
            visualize_condition_consistency(model, epoch+1, SAMPLE_DIR)  # New visualization
        
        # Save model checkpoint
        if (epoch + 1) % 500 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'vaegan3_revb_checkpoint_epoch_{epoch+1}')
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Print time elapsed
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")
    
    # Save final model
    print("Training complete! Saving final model...")
    
    # Create and trace a concrete function to ensure the model is properly serialized
    # First get sample data
    for batch in dataset.take(1):
        # In this dataset, the batch is already a dictionary with 'images' and 'conditions' keys
        sample_images = batch['images']
        sample_conditions = batch['conditions']
        break
    
    # Define a function for tracing
    @tf.function
    def inference_fn(images, conditions):
        return model({'images': images, 'conditions': conditions})
    
    # Trace the function with concrete input shapes
    inference_fn = inference_fn.get_concrete_function(
        tf.TensorSpec(sample_images.shape, tf.float32),
        tf.TensorSpec(sample_conditions.shape, tf.float32)
    )
    
    # Create a SavedModel
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'vaegan3_revb_final')
    tf.saved_model.save(
        model, 
        final_model_path,
        signatures={'serving_default': inference_fn}
    )
    print(f"Saved final model to {final_model_path}")
    
    # Also save a checkpoint version if needed for compatibility
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'vaegan3_revb_final_ckpt')
    model.save_weights(checkpoint_path)
    print(f"Saved model weights to {checkpoint_path}")
    
    print("Training completed!")

if __name__ == '__main__':
    main() 