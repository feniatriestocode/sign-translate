import tensorflow as tf

# Load the saved Keras model
model = tf.keras.models.load_model("../models/asl_model_norm.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("../models/asl_model_norm.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as asl_model_norm.tflite")