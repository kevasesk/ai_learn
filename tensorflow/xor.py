import tensorflow as tf
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float);

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float);

# Check if the model is already saved
try:
    model =  tf.keras.models.load_model("xor_model.keras")
except (OSError, ValueError):
    model = None

if model is None:
    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=700, verbose=1)

    model.save("xor_model.keras")
# Load the model


# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X)

# Print the predictions
for i in range(4):
    print(f"Input: {X[i]}, True Output: {y[i][0]}, Predicted Output: {predictions[i][0]:.4f}")

# Print the predicted classes
predicted_classes = [1 if pred[0] > 0.5 else 0 for pred in predictions]
for i in range(4):
    print(f"Input: {X[i]}, True Output: {y[i][0]}, Predicted Class: {predicted_classes[i]}")


weights_layer1 = model.layers[0].get_weights()
print("First Layer Weights (Kernel):")
print(weights_layer1[0])  # Shape (2, 2)
print("First Layer Biases:")
print(weights_layer1[1])  # Shape (2,)

# Get weights of the second layer
weights_layer2 = model.layers[1].get_weights()
print("Second Layer Weights (Kernel):")
print(weights_layer2[0])  # Shape (2, 1)
print("Second Layer Biases:")
print(weights_layer2[1])  # Shape (1,)