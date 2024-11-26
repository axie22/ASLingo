from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2

def get_model(input_shape=(64, 64, 3), num_classes=36):
    """
    Creates and compiles the MobileNetV2-based model for ASL classification.

    Args:
        input_shape (tuple): Shape of input images, default is (64, 64, 3).
        num_classes (int): Number of classes for classification, default is 36.

    Returns:
        keras.models.Model: Compiled Keras model.
    """
    # Load MobileNetV2 with pre-trained weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 36 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
