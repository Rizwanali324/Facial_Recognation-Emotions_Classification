import tensorflow as tf

model = tf.keras.models.load_model('code/models/emotiondetector.h5')

# Get the input shape of the model
input_shape = model.input_shape[1:]  # Exclude the batch size

# Print the input shape for verification
print(f"Model input shape: {input_shape}")

height, width, channels = input_shape

# Create a TFLiteConverter object from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization flag to optimize for size and latency
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define a representative dataset generator
def representative_dataset_gen():
    for _ in range(100):
        yield [tf.random.normal([1, height, width, channels], dtype=tf.float32)]

# Assign the representative dataset generator to the converter
converter.representative_dataset = representative_dataset_gen

# Perform the conversion
tflite_model = converter.convert()

# Save the optimized TFLite model
tflite_model_path = 'code/models/emotiondetector_optimized.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Optimized model has been successfully converted to TFLite and saved as {tflite_model_path}.")
