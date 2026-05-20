import tensorflow as tf
import numpy as np
import cv2
import albumentations as A

class bird_loader:
    def __init__(self, image_paths, labels, is_train=True, batch_size=32):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.batch_size = batch_size
        
        # define augmentation pipeline using albumentations
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
            ])
            
    def _process_image(self, image_path, label):
        img_path_str = image_path.numpy().decode("utf-8")
        image = cv2.imread(img_path_str)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply augmentations
        augmented = self.transform(image=image)
        image = augmented["image"].astype(np.float32)
        
                
        return image, np.int64(label.numpy())
    
    def _tf_wrapper(self, image_path, label):
        image, label = tf.py_function(self._process_image, [image_path, label], [tf.float32, tf.int64])
        image.set_shape((224, 224, 3))
        label.set_shape([])
        return image, label
    
    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))
        
        dataset = dataset.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
