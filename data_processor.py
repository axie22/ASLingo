from keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(64, 64), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the dataset using ImageDataGenerator.
    
    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target size for the images.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of data to use for validation.
    
    Returns:
        train_data: Training data generator.
        val_data: Validation data generator.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2
    )
    
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    print("TRAIN DATA INDICIES")
    print(train_data.class_indices)
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_data, val_data
