import os
import pandas as pd

def prepare_data(data_dir="data/raw/CUB_200_2011/CUB_200_2011", output_dir="data/processed"):
   """
   parses raw CUB-200 text files and splits images into clean train/val CSV files.
   """
   print("preparing data...")
   
   # 1. define paths to raw data files
   images_txt = os.path.join(data_dir, "images.txt")
   labels_txt = os.path.join(data_dir, "image_class_labels.txt")
   split_txt = os.path.join(data_dir, "train_test_split.txt")
   classes_txt = os.path.join(data_dir, "classes.txt")
    
    # verify the raw dataset exists before running
   if not os.path.exists(images_txt):
       raise FileNotFoundError(f"dataset files not found at {data_dir}. please download and extract the CUB-200-2011 dataset from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")
    
    # 2. read the text files into pandas dataframes
   print("reading raw data files...")
    
   df_images = pd.read_csv(images_txt, sep=r"\s+", names=["image_id", "filepath"])
   df_labels = pd.read_csv(labels_txt, sep=r"\s+", names=["image_id", "class_id"])
   df_split = pd.read_csv(split_txt, sep=r"\s+", names=["image_id", "is_training_img"])
   df_classes = pd.read_csv(classes_txt, sep=r"\s+", names=["class_id", "class_name"])
    
    # 3. merge together into a single dataframe
   print("merging dataframes...")
   df = df_images.merge(df_labels, on="image_id")
   df = df.merge(df_split, on="image_id")
   df = df.merge(df_classes, on="class_id")
    
    # shift class_id to be 0-indexed
   df["label"] = df["class_id"] - 1
    
    # reconstruct the absolute local path for each image file
   imgs_folder_path = os.path.join(data_dir, "images")
   df["full_path"] = df["filepath"].apply(lambda x: os.path.normpath(os.path.join(imgs_folder_path, x)))
    
    # 4. split into train and val dataframes
   train_df = df[df["is_training_img"] == 1][["full_path", "label", "class_name"]]
   val_df = df[df["is_training_img"] == 0][["full_path", "label", "class_name"]]
    
    # 5. save to output directory as CSV files
   os.makedirs(output_dir, exist_ok=True)
    
   train_csv_path = os.path.join(output_dir, "train.csv")
   val_csv_path = os.path.join(output_dir, "val.csv")
    
   train_df.to_csv(train_csv_path, index=False)
   val_df.to_csv(val_csv_path, index=False)
    
   print(f"data preparation complete!")
   print(f"train data saved to: {train_csv_path} ({len(train_df)} images)")
   print(f"val data saved to: {val_csv_path} ({len(val_df)} images)")
    
if __name__ == "__main__":
   prepare_data()