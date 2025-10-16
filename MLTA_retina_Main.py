import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import pywt

# Configuration
CONFIG = {
    'img_height': 300,
    'img_width': 300,
    'channels': 3,
    'num_classes': 4,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'weight_decay': 0.01,
    'embed_dim': 256,
    'num_heads': 8,
    'ff_dim': 512,
    'transformer_layers': 6,
    'throttling_lambda': 0.5
}

class Preprocessing:
    """Dual-domain preprocessing with spatial and frequency domains"""
    
    def __init__(self):
        self.denoise_model = self._build_denoise_model()
    
    def _build_denoise_model(self):
        """Lightweight CNN for denoising"""
        inputs = keras.Input(shape=(None, None, 3))
        
        # Encoder
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
     
        noise = layers.Conv2D(3, 3, activation='tanh', padding='same')(x)
        denoised = layers.Subtract()([inputs, noise])
        
        return keras.Model(inputs, denoised)
    
    def preprocess_batch(self, images):
        """Preprocess batch of images"""
        # Simple normalization for now - you can add more complex preprocessing
        return tf.cast(images, tf.float32) / 255.0


class ThrottledAttention(layers.Layer):
    """Throttled Attention Mechanism"""
    def __init__(self, embed_dim, num_heads, throttling_lambda=0.5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.throttling_lambda = throttling_lambda

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
     
        self.importance_dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
   
        attended = self.attention(inputs, inputs)  # (batch, seq_len, embed_dim)

    
        importance_scores = self.importance_dense(attended)  # (batch, seq_len, 1)
        importance_scores = tf.squeeze(importance_scores, axis=-1)  # (batch, seq_len)

        if training:
          
            importance_scores = tf.cast(importance_scores, tf.float32)
            mean_scores = tf.reduce_mean(importance_scores, axis=1, keepdims=True)  # (batch,1)
            std_scores = tf.math.reduce_std(importance_scores, axis=1, keepdims=True)  # (batch,1)
            threshold = mean_scores + self.throttling_lambda * std_scores  # (batch,1)

            mask = tf.cast(importance_scores > threshold, tf.float32)  # (batch, seq_len)
            mask_exp = tf.expand_dims(mask, axis=-1)  # (batch, seq_len, 1)

            masked_attended = attended * mask_exp  # zeros out low-importance tokens

          
            sum_kept = tf.reduce_sum(masked_attended, axis=1, keepdims=True)  # (batch,1,embed_dim)
            total_sum = tf.reduce_sum(attended, axis=1, keepdims=True)        # (batch,1,embed_dim)

            # compute reweighting factor per feature-dim
            reweighting_factor = total_sum / (sum_kept + 1e-8)  # (batch,1,embed_dim)
            throttled_output = masked_attended * reweighting_factor

            # keep same dtype as inputs
            return tf.cast(throttled_output, attended.dtype)
        else:
            return attended

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'throttling_lambda': self.throttling_lambda
        })
        return config


class PositionalEncoding(layers.Layer):
    """Simplified, correct Positional Encoding using tf ops"""
    def __init__(self, max_positions, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = int(max_positions)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        position = tf.cast(tf.range(self.max_positions)[:, tf.newaxis], tf.float32)  # (max_positions, 1)
        i = tf.cast(tf.range(self.embed_dim)[tf.newaxis, :], tf.float32)             # (1, embed_dim)

     
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = position * angle_rates  # (max_positions, embed_dim)

        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])

    
        pos_encoding = tf.zeros((self.max_positions, self.embed_dim), dtype=tf.float32)
        pos_encoding = tf.reshape(
            tf.stack([sines, coses], axis=-1),
            (self.max_positions, -1)
        )

     
        pos_encoding = pos_encoding[:, :self.embed_dim]
        self.pos_encoding = tf.expand_dims(pos_encoding, 0)  # (1, max_positions, embed_dim)

        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        base = super().get_config()
        base.update({'max_positions': self.max_positions, 'embed_dim': self.embed_dim})
        return base



class MultiLevelFeatureExtractor(layers.Layer):
    """Multi-level feature extraction from EfficientNet"""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Build EfficientNet backbone

        self.backbone = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights='imagenet',
            input_shape=(config['img_height'], config['img_width'], config['channels'])
        )
        
        # Get intermediate layers for multi-level features
        self.feature_layers = [
            'block2a_expand_activation',  # Low-level features
            'block4a_expand_activation',  # Intermediate features  
            'block6a_expand_activation',  # Mid-level features
            'top_activation'              # High-level features
        ]
        
        # Create feature extractor model
        layer_outputs = [self.backbone.get_layer(name).output for name in self.feature_layers]
        self.feature_extractor = keras.Model(inputs=self.backbone.input, outputs=layer_outputs)
        
        # Make backbone non-trainable for feature extraction
        self.backbone.trainable = False
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        return features

class VisionTransformerBlock(layers.Layer):
    """Vision Transformer Block with Throttled Attention"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, throttling_lambda=0.5, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.throttling_lambda = throttling_lambda
        self.rate = rate
        
        self.att = ThrottledAttention(embed_dim, num_heads, throttling_lambda)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'throttling_lambda': self.throttling_lambda,
            'rate': self.rate
        })
        return config


def build_mlta_model(config):
  
    
    # Input
    inputs = keras.Input(shape=(config['img_height'], config['img_width'], config['channels']))
    
    x = layers.Rescaling(1./255)(inputs)
    
    # Multi-level feature extraction
    feature_extractor = MultiLevelFeatureExtractor(config)
    features = feature_extractor(x)
    
    # Process each feature level with throttled attention
    processed_features = []
    for i, feature in enumerate(features):
        # Global average pooling
        x_level = layers.GlobalAveragePooling2D()(feature)
        x_level = layers.Dense(config['embed_dim'], activation='relu')(x_level)
        x_level = layers.Reshape((1, config['embed_dim']))(x_level)
        
        # Throttled attention
        throttled_att = ThrottledAttention(
            config['embed_dim'], 
            config['num_heads'],
            config['throttling_lambda']
        )
        x_level = throttled_att(x_level)
        processed_features.append(x_level)
    
    # Concatenate all processed features
    cnn_features = layers.Concatenate(axis=1)(processed_features)
    
    # Positional encoding
    positional_encoding = PositionalEncoding(
        max_positions=len(features),  # 4 feature levels
        embed_dim=config['embed_dim']
    )
    x = positional_encoding(cnn_features)
    
    # Transformer layers
    for i in range(config['transformer_layers']):
        transformer_block = VisionTransformerBlock(
            config['embed_dim'],
            config['num_heads'],
            config['ff_dim'],
            config['throttling_lambda']
        )
        x = transformer_block(x)
    
    # Global average pooling
    transformer_output = layers.GlobalAveragePooling1D()(x)
    
    # Additional CNN global features
    cnn_global = layers.GlobalAveragePooling2D()(feature_extractor.backbone(inputs))
    cnn_global = layers.Dense(config['embed_dim'], activation='relu')(cnn_global)
    
    # Fusion
    fused = layers.Concatenate()([transformer_output, cnn_global])
    fused = layers.Dense(512, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    
    # Output
    outputs = layers.Dense(config['num_classes'], activation='softmax')(fused)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLTA_Model")
    return model

class OCTDataLoader:
    """Data loader for OCT images"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocessing()
    
    def load_data(self, data_dir, validation_split=0.2):
        """Load and preprocess OCT dataset"""
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(self.config['img_height'], self.config['img_width']),
            batch_size=self.config['batch_size'],
            label_mode='categorical'
        )
        
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(self.config['img_height'], self.config['img_width']),
            batch_size=self.config['batch_size'],
            label_mode='categorical'
        )
        
        # Apply preprocessing
        train_dataset = train_dataset.map(
            lambda x, y: (self.preprocessor.preprocess_batch(x), y)
        )
        val_dataset = val_dataset.map(
            lambda x, y: (self.preprocessor.preprocess_batch(x), y)
        )
        
        # Optimize dataset performance
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_mlta_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    return callbacks

def compile_model(model, config):
    """Compile the model"""
    optimizer = keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

def evaluate_model(model, test_dataset):
    """Comprehensive model evaluation"""
    # Get true labels and predictions
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in test_dataset:
        y_true.extend(y_batch.numpy())
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(preds)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Evaluate
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset, verbose=0)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'confusion_matrix': cm
    }

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    print("Initializing MLTA Model...")
    
    # Build model
    model = build_mlta_model(CONFIG)
    
    print("Model architecture:")
    model.summary()
    
    # Load data
    print("Loading data...")
    data_loader = OCTDataLoader(CONFIG)
    
    dataset_path = '/content/OCTID' 
    try:
        train_dataset, val_dataset = data_loader.load_data(dataset_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please update the dataset_path in main() function")
        return None, None, None
  
    model = compile_model(model, CONFIG)
    

    callbacks = create_callbacks()
 
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=CONFIG['epochs'],
        validation_data=val_dataset,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, val_dataset)
    

    return model, history, results

def simple_train():
    """Simplified training function"""
    print("Building MLTA Model...")
    model = build_mlta_model(CONFIG)
    

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

if __name__ == "__main__":
 
    model, history, results = main()
    
    if model is None:
        print("Trying simple training...")
        model = simple_train()