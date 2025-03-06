import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, initializers, metrics, optimizers, losses

# Default parameters for the model
LATENT_DIM = 1
CONDITION_DIM = 15
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

class VAEGAN3(Model):
    """
    Enhanced Conditional Variational Autoencoder + Generative Adversarial Network hybrid model.
    Combines VAE's ability to learn smooth latent spaces with GAN's capability for 
    generating realistic images, plus conditional consistency to ensure better interpolation.
    
    This model extends the VAEGAN2 by adding a contrastive loss component
    that helps prevent different conditions from mapping to the same output space.
    The contrastive loss encourages samples with different conditions to have
    distinct representations in the latent space.
    """
    
    def __init__(self, 
                 latent_dim=LATENT_DIM, 
                 condition_dim=CONDITION_DIM,
                 image_size=IMAGE_SIZE,
                 image_channels=IMAGE_CHANNELS,
                 consistency_weight=1.0,
                 contrastive_weight=1.0,
                 temperature=0.5):
        """
        Initialize the enhanced VAE-GAN model with conditional consistency and contrastive loss.
        
        Args:
            latent_dim: Dimension of the latent space
            condition_dim: Dimension of the condition vector
            image_size: Size of the input/output images (assumed square)
            image_channels: Number of channels in the images
            consistency_weight: Weight for the conditional consistency loss
            contrastive_weight: Weight for the contrastive loss
            temperature: Temperature parameter for contrastive loss
        """
        super(VAEGAN3, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.consistency_weight = consistency_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # Initialize training epoch counter as a TensorFlow variable
        self._training_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="training_epoch")
        
        # Build the networks
        self.encoder = self._build_encoder()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.condition_predictor = self._build_condition_predictor()  # Network for condition prediction
        
        # Compile optimizers
        self.vae_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00002, beta_1=0.1)
        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
        self.predictor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Setup metrics to track
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.vae_loss_tracker = metrics.Mean(name="vae_loss")
        self.gen_loss_tracker = metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = metrics.Mean(name="discriminator_loss")
        self.consistency_loss_tracker = metrics.Mean(name="consistency_loss")
        self.contrastive_loss_tracker = metrics.Mean(name="contrastive_loss")  # Additional metric for contrastive loss
        self.total_loss_tracker = metrics.Mean(name="total_loss")
    
    @property
    def metrics(self):
        """Return metrics for Keras API."""
        return [
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.vae_loss_tracker,
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.consistency_loss_tracker,
            self.contrastive_loss_tracker,
            self.total_loss_tracker,
        ]
    
    def compile(self, vae_optimizer=None, discriminator_optimizer=None, 
                generator_optimizer=None, predictor_optimizer=None, **kwargs):
        """Compile the model with optimizers."""
        super(VAEGAN3, self).compile(**kwargs)
        
        # Use provided optimizers or default ones
        self.vae_optimizer = vae_optimizer or self.vae_optimizer
        self.discriminator_optimizer = discriminator_optimizer or self.discriminator_optimizer
        self.generator_optimizer = generator_optimizer or self.generator_optimizer
        self.predictor_optimizer = predictor_optimizer or self.predictor_optimizer
    
    def _build_encoder(self):
        """Build the encoder network following the REVB VAE architecture."""
        # Image input
        image_input = layers.Input(shape=(self.image_size, self.image_size, self.image_channels), name="encoder_image_input")
        
        # Condition input
        condition_input = layers.Input(shape=(self.condition_dim,), name="encoder_condition_input")
        
        # Encode the image - using the same architecture as CVAE_REVB
        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image_input)
        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        
        # Process condition and concatenate
        c = layers.Dense(64, activation="relu")(condition_input)
        
        x = layers.Concatenate()([x, c])
        
        # Fully connected layer
        x = layers.Dense(128, activation="relu")(x)
        
        # Output mean and log variance
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Create the encoder model
        encoder_model = Model([image_input, condition_input], [z_mean, z_log_var], name="encoder")
        return encoder_model
    
    def _build_generator(self):
        """Build the generator/decoder network following the REVB models."""
        # Latent space input
        latent_input = layers.Input(shape=(self.latent_dim,), name="generator_latent_input")
        
        # Condition input
        condition_input = layers.Input(shape=(self.condition_dim,), name="generator_condition_input")
        
        # Combine latent and condition
        x = layers.Concatenate()([latent_input, condition_input])
        
        # Dense layer to get to spatial dimensions - same as REVB models
        x = layers.Dense(16 * 16 * 64, activation="relu")(x)
        x = layers.Reshape((16, 16, 64))(x)
        
        # First upsampling block
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        
        # Second upsampling block
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(self.image_channels, 3, padding="same", activation="sigmoid")(x)
        
        # Create the generator model
        generator_model = Model([latent_input, condition_input], x, name="generator")
        return generator_model
    
    def _build_discriminator(self):
        """Build the discriminator network with appropriate scale."""
        # Image input
        image_input = layers.Input(shape=(self.image_size, self.image_size, self.image_channels), name="discriminator_image_input")
        
        # Condition input
        condition_input = layers.Input(shape=(self.condition_dim,), name="discriminator_condition_input")
        
        # Process the image with convolutions - simpler architecture like REVB GAN
        x = layers.Conv2D(32, 3, strides=2, padding="same")(image_input)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Flatten
        x = layers.Flatten()(x)
        
        # Process condition
        c = layers.Dense(64)(condition_input)
        c = layers.LeakyReLU(0.2)(c)
        
        # Concatenate image features with condition
        x = layers.Concatenate()([x, c])
        
        # Dense layers
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Output validity
        x = layers.Dense(1, activation="sigmoid")(x)
        
        # Create the discriminator model
        discriminator_model = Model([image_input, condition_input], x, name="discriminator")
        return discriminator_model
    
    def _build_condition_predictor(self):
        """Build a network to predict conditions from generated images."""
        # Image input
        image_input = layers.Input(shape=(self.image_size, self.image_size, self.image_channels))
        
        # Use a similar architecture as the discriminator for the condition predictor
        x = layers.Conv2D(32, 3, strides=2, padding="same")(image_input)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Output layer to predict condition values
        x = layers.Dense(self.condition_dim)(x)
        
        # Create the condition predictor model
        predictor_model = Model(image_input, x, name="condition_predictor")
        return predictor_model
    
    @tf.function
    def call(self, inputs, training=None):
        """Forward pass through the model.
        
        Args:
            inputs: Can be either:
                - A dictionary with 'images' and 'conditions' keys
                - A tuple/list with [images, conditions]
                
        Returns:
            A dictionary with generated images and latent space parameters
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            images = inputs['images']
            conditions = inputs['conditions']
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            images, conditions = inputs
        else:
            raise ValueError("Inputs must be either a dictionary with 'images' and 'conditions' keys or a tuple/list with [images, conditions]")
        
        # Encode the images to get mean and log variance
        z_mean, z_log_var = self.encoder([images, conditions])
        
        # Sample from the latent space
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Generate images from the latent representation
        generated_images = self.generator([z, conditions])
        
        return {"generated_images": generated_images, "z_mean": z_mean, "z_log_var": z_log_var}
    
    def _compute_vae_loss(self, images, generated_images, z_mean, z_log_var):
        """Compute VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss (mean squared error)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(
                tf.reshape(images, [-1, self.image_size * self.image_size * self.image_channels]),
                tf.reshape(generated_images, [-1, self.image_size * self.image_size * self.image_channels])
            )
        )
        
        # KL divergence
        beta_vae_weight = 8.0
        kl_loss = beta_vae_weight *-0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        # Total VAE loss
        vae_loss = reconstruction_loss + kl_loss
        
        return vae_loss, reconstruction_loss, kl_loss
    
    def _compute_discriminator_loss(self, real_output, fake_output):
        """Compute discriminator loss using binary cross-entropy."""
        real_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
        )
        fake_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        )
        
        return real_loss + fake_loss
    
    def _compute_generator_loss(self, fake_output):
        """Compute generator loss using binary cross-entropy."""
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)
        )
    
    def _compute_consistency_loss(self, conditions, predicted_conditions):
        """
        Compute the consistency loss between original conditions and predicted conditions.
        This loss encourages the generator to produce images that match the input conditions.
        """
        # Cast both tensors to the same data type (float32) to avoid type mismatch
        conditions = tf.cast(conditions, tf.float32)
        predicted_conditions = tf.cast(predicted_conditions, tf.float32)
        return tf.reduce_mean(tf.square(conditions - predicted_conditions))
    
    def _compute_contrastive_loss(self, latents, conditions):
        """
        Compute contrastive loss to ensure different conditions map to different regions of latent space.
        
        This implementation uses a simplified contrastive loss approach where we:
        1. Compute pairwise distances between latents
        2. Compute pairwise distances between conditions
        3. Push latents apart when conditions are different
        4. Pull latents together when conditions are similar
        
        Args:
            latents: Batch of latent vectors (z_mean)
            conditions: Batch of condition vectors
            
        Returns:
            Contrastive loss value
        """
        batch_size = tf.shape(latents)[0]
        
        # Normalize latents and conditions
        latents_norm = tf.math.l2_normalize(latents, axis=1)
        conditions_norm = tf.math.l2_normalize(conditions, axis=1)
        
        # Compute similarity matrix for latents
        latent_sim = tf.matmul(latents_norm, latents_norm, transpose_b=True) / self.temperature
        
        # Compute similarity matrix for conditions
        condition_sim = tf.matmul(conditions_norm, conditions_norm, transpose_b=True)
        
        # Create mask to identify similar/dissimilar pairs
        # Higher values in condition_sim mean more similar conditions
        # We use a threshold of 0.8 to identify "same condition" pairs
        # This could be tuned based on your specific condition space
        similarity_threshold = 0.8
        positive_mask = tf.cast(condition_sim > similarity_threshold, tf.float32)
        
        # Account for self-similarity
        mask = tf.eye(batch_size)
        positive_mask = positive_mask - mask
        
        # Compute contrastive loss
        # For each anchor, compute its attraction to positives and repulsion from negatives
        exp_logits = tf.exp(latent_sim)
        
        # Remove diagonal from numerator (self-similarity)
        exp_logits_without_self = exp_logits * (1 - mask)
        
        # For each anchor, sum similarities with positives
        pos_sum = tf.reduce_sum(exp_logits_without_self * positive_mask, axis=1)
        
        # For each anchor, sum similarities with all other examples
        all_sum = tf.reduce_sum(exp_logits_without_self, axis=1)
        
        # Compute final loss: -log(pos_sum / all_sum)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        contrastive_loss = -tf.reduce_mean(tf.math.log((pos_sum + epsilon) / (all_sum + epsilon)))
        
        return contrastive_loss
    
    @tf.function(input_signature=None)
    def train_step(self, data):
        # Use tf.print instead of Python print for graph mode execution
        tf.print("Training epoch:", self._training_epoch)
        
        # Print training phase using tf.cond for graph-mode compatibility
        phase = tf.cond(
            self._training_epoch < 1000 or self._training_epoch % 2 == 0, # we actually alternate vae and gan every epoch after 1000
            lambda: "Phase 1: VAE-only training",
            lambda: "Phase 2: Full GAN training"
        )
        tf.print(phase)
        
        """Train the model for one step with conditional consistency and contrastive loss."""
        images = data['images']
        conditions = data['conditions']
        
        # Combined training step with all sub-networks
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as vae_tape, tf.GradientTape() as pred_tape:
            # Get output from the encoder
            z_mean, z_log_var = self.encoder([images, conditions], training=True)
            
            # Sample from the latent distribution
            batch_size = tf.shape(z_mean)[0]
            epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            # Generate images from the latent representation
            generated_images = self.generator([z, conditions], training=True)
            
            # Compute VAE loss (reconstruction + KL)
            vae_loss, reconstruction_loss, kl_loss = self._compute_vae_loss(images, generated_images, z_mean, z_log_var)
            
            # Process based on training phase using tf.cond for graph-mode compatibility
            def phase1():
                # Phase 1: Only VAE loss
                return (
                    tf.constant(0.0),  # disc_loss
                    tf.constant(0.0),  # gen_loss
                    tf.constant(0.0),  # consistency_loss
                    tf.constant(0.0),  # contrastive_loss
                    vae_loss,          # total_vae_loss
                    vae_loss           # total_loss
                )
                
            def phase2():
                # Phase 2: All losses active
                # Get discriminator outputs
                real_output = self.discriminator([images, conditions], training=True)
                fake_output = self.discriminator([generated_images, conditions], training=True)
                
                # Compute discriminator and generator losses
                disc_loss = self._compute_discriminator_loss(real_output, fake_output)
                gen_loss = self._compute_generator_loss(fake_output)
                
                # Get predicted conditions from the condition predictor
                predicted_conditions = self.condition_predictor(generated_images, training=True)
                
                # Compute condition consistency loss
                consistency_loss = self._compute_consistency_loss(conditions, predicted_conditions)
                
                # Compute contrastive loss
                contrastive_loss = self._compute_contrastive_loss(z, conditions)
                
                # Compute total losses
                total_vae_loss = vae_loss + 0.5 * gen_loss + self.consistency_weight * consistency_loss + self.contrastive_weight * contrastive_loss
                total_loss = vae_loss + 0.5 * gen_loss + self.consistency_weight * consistency_loss + self.contrastive_weight * contrastive_loss
                
                return (
                    disc_loss,
                    gen_loss,
                    consistency_loss,
                    contrastive_loss,
                    total_vae_loss,
                    total_loss
                )
            
            # Use tf.cond to conditionally execute based on training phase
            disc_loss, gen_loss, consistency_loss, contrastive_loss, total_vae_loss, total_loss = tf.cond(
                self._training_epoch < 1000 or self._training_epoch % 2 == 0, # we actually alternate vae and gan every epoch after 1000
                phase1,
                phase2
            )
        
        # Get VAE gradients always
        vae_gradients = vae_tape.gradient(total_vae_loss, self.encoder.trainable_variables + self.generator.trainable_variables)
        encoder_vars = self.encoder.trainable_variables
        generator_vars = self.generator.trainable_variables
        
        # Apply VAE gradients always
        self.vae_optimizer.apply_gradients(zip(
            vae_gradients[:len(encoder_vars)], encoder_vars
        ))
        self.vae_optimizer.apply_gradients(zip(
            vae_gradients[len(encoder_vars):], generator_vars
        ))
        
        # Apply GAN-related gradients conditionally
        def apply_gan_gradients():
            # Get discriminator and predictor gradients
            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            pred_gradients = pred_tape.gradient(consistency_loss, self.condition_predictor.trainable_variables)
            
            # Apply discriminator gradients  
            self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
            
            # Apply predictor gradients
            self.predictor_optimizer.apply_gradients(zip(pred_gradients, self.condition_predictor.trainable_variables))
            return tf.constant(1.0)  # Dummy return
            
        def skip_gan_gradients():
            return tf.constant(0.0)  # Dummy return
            
        # Conditionally apply GAN gradients
        _ = tf.cond(
            self._training_epoch >= 1000,
            apply_gan_gradients,
            skip_gan_gradients
        )
        
        # Update metrics
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.vae_loss_tracker.update_state(vae_loss)
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        self.consistency_loss_tracker.update_state(consistency_loss)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "vae_loss": self.vae_loss_tracker.result(),
            "generator_loss": self.gen_loss_tracker.result(),
            "discriminator_loss": self.disc_loss_tracker.result(),
            "consistency_loss": self.consistency_loss_tracker.result(),
            "contrastive_loss": self.contrastive_loss_tracker.result(),
        }
    
    def generate(self, conditions):
        """Generate images from conditions by sampling from latent space."""
        batch_size = tf.shape(conditions)[0]
        
        # Generate random latent vectors
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Generate images
        return self.generator([z, conditions])
    
    def generate_with_latent(self, latent_vectors, conditions):
        """Generate images from specified latent vectors and conditions."""
        return self.generator([latent_vectors, conditions])
    
    def get_config(self):
        """Return the configuration of the model for serialization."""
        config = super(VAEGAN3, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'image_size': self.image_size, 
            'image_channels': self.image_channels,
            'consistency_weight': self.consistency_weight,
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create a model from its config."""
        return cls(**config)

    # Add getter and setter properties for training_epoch
    @property
    def training_epoch(self):
        """Get the current training epoch."""
        return self._training_epoch.numpy()
    
    @training_epoch.setter
    def training_epoch(self, value):
        """Set the training epoch."""
        print(f"Setting _training_epoch to {value}")
        self._training_epoch.assign(value)


def get_vaegan3_model(latent_dim=LATENT_DIM, 
                     condition_dim=CONDITION_DIM, 
                     image_size=IMAGE_SIZE, 
                     image_channels=IMAGE_CHANNELS,
                     consistency_weight=1.0,
                     contrastive_weight=1.0,
                     temperature=0.5):
    """Factory function to create and initialize a VAEGAN3 model."""
    model = VAEGAN3(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        image_size=image_size,
        image_channels=image_channels,
        consistency_weight=consistency_weight,
        contrastive_weight=contrastive_weight,
        temperature=temperature
    )
    
    # Compile the model with default optimizers
    model.compile()
    
    return model


def load_vaegan3_model(model_path):
    """
    Load a saved VAEGAN3 model.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        A function that generates images from conditions
    """
    # Load the model with custom objects
    loaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'VAEGAN3': VAEGAN3}
    )
    
    # Create a generator function with a simpler interface
    def generate_images(conditions, n_samples=1):
        """
        Generate images for given conditions.
        
        Args:
            conditions: A numpy array of condition vectors, shape [batch_size, condition_dim]
            n_samples: Number of samples to generate per condition
            
        Returns:
            A batch of generated images
        """
        conditions = tf.convert_to_tensor(conditions, dtype=tf.float32)
        batch_size = tf.shape(conditions)[0]
        
        # Generate random latent vectors
        latent_vectors = tf.random.normal(shape=(batch_size * n_samples, loaded_model.latent_dim))
        
        # Repeat conditions for each sample
        repeated_conditions = tf.repeat(conditions, repeats=n_samples, axis=0)
        
        # Generate images
        return loaded_model.generator([latent_vectors, repeated_conditions])
    
    return generate_images


def export_generator_for_tfjs(model_path, export_path):
    """
    Export only the generator part of the model for TensorFlow.js.
    
    Args:
        model_path: Path to the full saved model
        export_path: Path to save the generator-only model
    """
    # Load the full model
    full_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'VAEGAN3': VAEGAN3}
    )
    
    # Extract the generator
    generator = full_model.generator
    
    # Save the generator model
    generator.save(export_path)
    
    return generator 