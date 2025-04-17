from model import get_model
from data_processor import load_data
import matplotlib.pyplot as plt

data_dir    = 'asl_dataset'
img_size    = (64, 64)
batch_size  = 32
val_split   = 0.2
test_split  = 0.1

def plot_history(history):
    epochs = range(1, len(history.history['loss'])+1)

    plt.figure()
    plt.plot(epochs, history.history['loss'],    label='Train loss')
    plt.plot(epochs, history.history['val_loss'],label='Val   loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history['accuracy'],    label='Train acc')
    plt.plot(epochs, history.history['val_accuracy'],label='Val   acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


train_data, val_data, test_data = load_data(
    data_dir, img_size, batch_size, val_split, test_split
)

model = get_model(input_shape=img_size + (3,), num_classes=36)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


loss, acc = model.evaluate(test_data, verbose=1)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4%}")

model.save('asl_model.h5')
model.summary()

# visualize training and validation curves
plot_history(history)