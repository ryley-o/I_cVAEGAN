import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
import json
import numpy as np
from PIL import Image
from vaegan3_revb import VAEGAN3, LATENT_DIM, CONDITION_DIM, IMAGE_SIZE, IMAGE_CHANNELS

def create_generator_model():
    """Create a standalone generator model from the trained VAE-GAN3 model."""
    # Load the trained VAE-GAN3 model
    vaegan3 = VAEGAN3(latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM)
    
    # Create dummy inputs to build the model
    dummy_images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
    dummy_conditions = np.zeros((1, CONDITION_DIM))
    
    # Call the model once to build it
    _ = vaegan3({"images": dummy_images, "conditions": dummy_conditions})
    
    # Load weights
    try:
        vaegan3.load_weights('models_revb/vaegan3_revb_final')
        print("Successfully loaded weights from models_revb/vaegan3_revb_final")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

    # Now extract the generator from the VAE-GAN3 model
    original_generator = vaegan3.generator
    print("\nOriginal generator model summary:")
    original_generator.summary()
    
    # Directly extract the weights we need from the original model
    dense_weights = None
    conv2d_weights = None
    conv2d_1_weights = None
    
    for layer in original_generator.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_weights = layer.get_weights()
            print(f"Found Dense layer with weights shapes: {[w.shape for w in dense_weights]}")
        elif isinstance(layer, tf.keras.layers.Conv2D) and ('conv2d_2' in layer.name or 'conv2d' in layer.name):
            if conv2d_weights is None:
                conv2d_weights = layer.get_weights()
                print(f"Found first Conv2D layer with weights shapes: {[w.shape for w in conv2d_weights]}")
            else:
                conv2d_1_weights = layer.get_weights()
                print(f"Found second Conv2D layer with weights shapes: {[w.shape for w in conv2d_1_weights]}")
    
    if not dense_weights or not conv2d_weights or not conv2d_1_weights:
        raise ValueError("Could not find all required weights in the original model")
    
    # Create a new model with the same architecture but single input
    inputs = tf.keras.layers.Input(shape=(LATENT_DIM + CONDITION_DIM,), name='combined_input')  # latent + condition
    x = tf.keras.layers.Dense(16 * 16 * 64, activation='relu')(inputs)
    x = tf.keras.layers.Reshape((16, 16, 64))(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    outputs = tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    
    lambda_free_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tfjs_generator")
    
    print("\nNew Lambda-free model summary:")
    lambda_free_model.summary()
    
    # Transfer weights to the new model
    for i, layer in enumerate(lambda_free_model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Setting weights for Dense layer at index {i}")
            layer.set_weights(dense_weights)
        elif isinstance(layer, tf.keras.layers.Conv2D) and layer.name.startswith('conv2d'):
            if layer.output_shape[-1] == 32:
                print(f"Setting weights for first Conv2D layer at index {i}")
                layer.set_weights(conv2d_weights)
            else:
                print(f"Setting weights for second Conv2D layer at index {i}")
                layer.set_weights(conv2d_1_weights)
    
    # Test the new model with a sample input
    test_input = np.zeros((1, LATENT_DIM + CONDITION_DIM))
    
    # Test with the original model
    latent_part = test_input[:, :LATENT_DIM]
    condition_part = test_input[:, LATENT_DIM:]
    original_output = original_generator([latent_part, condition_part])
    print(f"Original model test output shape: {original_output.shape}")
    
    # Test with the new Lambda-free model
    new_output = lambda_free_model(test_input)
    print(f"New Lambda-free model test output shape: {new_output.shape}")
    
    # Compare outputs to ensure the weights were transferred correctly
    mean_diff = tf.reduce_mean(tf.abs(original_output - new_output))
    print(f"Mean absolute difference between outputs: {mean_diff.numpy()}")
    if mean_diff.numpy() > 0.1:
        print("Warning: Outputs differ significantly. Weight transfer may not be perfect.")
    
    return lambda_free_model

def save_model_json(model, output_dir):
    """Save model in a simple JSON format with weight quantization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model config and weights
    model_json = model.to_json()
    model_config = json.loads(model_json)
    
    # Save weights
    weight_manifest = []
    weight_idx = 0
    
    # Map layer names to their weights
    for layer in model.layers:
        if not layer.weights:
            continue
            
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
            
        # Create manifest entry for this layer
        layer_manifest = {
            'paths': [f'group{weight_idx}-shard1of1.bin'],
            'weights': []
        }
        
        # Save weights for this layer
        weight_data = []
        for i, w in enumerate(layer_weights):
            # Compute min and max for quantization
            w_min = float(w.min())
            w_max = float(w.max())
            
            # Quantize to uint8
            w_range = w_max - w_min
            w_scale = w_range / 255.0 if w_range > 0 else 1.0
            w_quantized = np.round((w - w_min) / w_scale).astype(np.uint8)
            weight_data.append(w_quantized)
            
            # Format the layer name without slashes
            clean_layer_name = layer.name.replace('/', '_')
            weight_type = 'kernel' if i == 0 else 'bias'
            
            # Add metadata for dequantization
            layer_manifest['weights'].append({
                'name': f'{clean_layer_name}/{weight_type}',
                'shape': list(w.shape),
                'dtype': 'float32',
                'quantization': {
                    'min': w_min,
                    'scale': w_scale,
                    'max': w_max,
                    'dtype': 'uint8'
                }
            })
        
        # Save quantized weights
        weight_data = np.concatenate([w.flatten() for w in weight_data])
        weight_data.tofile(os.path.join(output_dir, f'group{weight_idx}-shard1of1.bin'))
        
        weight_manifest.append(layer_manifest)
        weight_idx += 1
    
    # Create model config
    config = {
        'format': 'layers-model',
        'generatedBy': 'TensorFlow.js v4.0.0',
        'convertedBy': 'TensorFlow.js Converter v4.0.0',
        'modelTopology': model_config,
        'weightsManifest': weight_manifest
    }
    
    # Save model.json
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(config, f, ensure_ascii=True)

def generate_test_image(model, output_dir):
    """Generate a test image and save it to verify the model works."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a random latent vector and a condition
    combined_input = np.zeros((1, LATENT_DIM + CONDITION_DIM))
    combined_input[0, 0] = np.random.normal()  # Random latent
    combined_input[0, LATENT_DIM + 0] = 1.0    # First attribute = 1.0
    
    # Generate an image
    image = model.predict(combined_input)
    
    # Save the image
    image = (image[0] * 255).astype(np.uint8)
    
    # Use PIL to save the image
    img = Image.fromarray(image)
    img.save(os.path.join(output_dir, 'test_generation.png'))
    print(f"Test image saved to {os.path.join(output_dir, 'test_generation.png')}")
    
    # Print debug info
    print(f"Test image shape: {image.shape}")
    print(f"Test image min/max values: {image.min()}/{image.max()}")

def create_model_info_file(latent_dim, condition_dim, output_dir):
    """Create a model info JSON file for the web interface."""
    model_info = {
        "format": "layers-model",
        "generatedBy": "manual_export",
        "modelPath": "tfjs/model.json",
        "inputShape": [1, latent_dim + condition_dim],
        "outputShape": [1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS],
        "inputName": "combined_input",
        "outputName": "StatefulPartitionedCall",
        "version": "1.0",
        "latentDim": latent_dim,
        "conditionDim": condition_dim,
        "modelType": "vaegan3",
        "features": {
            "contrastiveLoss": True,
            "consistencyRegularization": True
        },
        "attributes": [
            {"index": 0, "name": "attr1", "description": "Attribute 1"},
            {"index": 1, "name": "attr2", "description": "Attribute 2"},
            {"index": 2, "name": "attr3", "description": "Attribute 3"},
            {"index": 3, "name": "attr4", "description": "Attribute 4"},
            {"index": 4, "name": "attr5", "description": "Attribute 5"},
            {"index": 5, "name": "attr6", "description": "Attribute 6"},
            {"index": 6, "name": "attr7", "description": "Attribute 7"},
            {"index": 7, "name": "attr8", "description": "Attribute 8"},
            {"index": 8, "name": "attr9", "description": "Attribute 9"},
            {"index": 9, "name": "attr10", "description": "Attribute 10"},
            {"index": 10, "name": "attr11", "description": "Attribute 11"},
            {"index": 11, "name": "attr12", "description": "Attribute 12"},
            {"index": 12, "name": "attr13", "description": "Attribute 13"},
            {"index": 13, "name": "attr14", "description": "Attribute 14"},
            {"index": 14, "name": "attr15", "description": "Attribute 15"}
        ]
    }
    
    with open(os.path.join(os.path.dirname(output_dir), 'model_info.json'), 'w') as f:
        json.dump(model_info, f, ensure_ascii=True, indent=2)

def main():
    try:
        print("Creating VAE-GAN3 generator model...")
        generator_model = create_generator_model()
        
        # Create output directory
        output_dir = 'web_model_revb/tfjs_vaegan3'
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating test image...")
        generate_test_image(generator_model, output_dir)
        
        print("Saving model in TensorFlow.js format...")
        # First save as a TensorFlow SavedModel (for potential future use)
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        tf.saved_model.save(generator_model, saved_model_dir)
        print(f"Saved TensorFlow model to {saved_model_dir}")
        
        # Save in custom TensorFlow.js format (avoiding JAX issues)
        save_model_json(generator_model, os.path.join(output_dir, 'tfjs'))
        
        # Create model info file
        create_model_info_file(LATENT_DIM, CONDITION_DIM, os.path.join(output_dir, 'tfjs'))
        
        print("Model successfully exported to 'web_model_revb/tfjs_vaegan3/tfjs' directory")
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"Error during export process: {str(e)}")
        raise

if __name__ == '__main__':
    main() 