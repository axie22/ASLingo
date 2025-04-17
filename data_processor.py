import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(64, 64), batch_size=32, val_split=0.2, test_split=0.1, random_state=42):
    """
    Scans data_dir/<class_name>/*.jpg into a DataFrame,
    then does stratified train/val/test splits, and returns
    three ImageDataGenerator.flow_from_dataframe objects.
    """
    # build DataFrame of all image paths + labels
    records = []
    classes = sorted(os.listdir(data_dir))
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            if os.path.isfile(path):
                records.append({"filepath": path, "label": cls})
    df = pd.DataFrame(records)

    # split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=val_split + test_split,
        stratify=df["label"],
        random_state=random_state,
    )

    val_frac = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_frac,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
    )
    val_gen = datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    test_gen = datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
