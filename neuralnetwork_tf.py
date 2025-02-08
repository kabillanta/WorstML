import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to a range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images to row vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(784,)),  # Input layer
    Dense(10, activation='relu'),                      # Hidden layer
    Dense(10, activation='softmax')                    # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model (TensorFlow will automatically use the GPU if available)
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
