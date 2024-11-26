from model import get_model  # Assuming model.py contains your model definition
from data_processor import load_data

# Define dataset parameters
data_dir = 'asl_dataset'  # Path to your dataset folder
img_size = (64, 64)
batch_size = 32

# Load data
train_data, val_data = load_data(data_dir, img_size, batch_size)

# Get the model
model = get_model(input_shape=(64, 64, 3), num_classes=36)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save the trained model
model.save('asl_model.h5')
model.summary()