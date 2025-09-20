## Run this to rename files in A & B folders sequentially from 0000.tif to n.tif

# import os

# def rename_files_in_order(folder_path):
#     # Get all .tif files sorted alphabetically
#     files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".tif")])

#     for index, filename in enumerate(files):
#         # Generate new filename with 4-digit zero padding
#         new_filename = f"{index:04d}.tif"

#         # Full paths
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)

#         # Rename file
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} --> {new_filename}")

#     print(f"✅ Renaming completed for folder: {folder_path}\n")


# # Update your folders here
# folder_A = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\A"
# folder_B = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\B"

# # Rename files in both folders
# rename_files_in_order(folder_A)
# rename_files_in_order(folder_B)


## Run this to rename files in label folder sequentially from 0000.tif to n.tif
# import os

# # Update your label folder path here
# label_folder = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\label"

# # Get all .tif files sorted alphabetically
# label_files = sorted([f for f in os.listdir(label_folder) if f.lower().endswith(".tif")])

# # Rename labels sequentially
# for index, filename in enumerate(label_files):
#     # Generate new filename with 4-digit zero padding
#     new_filename = f"{index:04d}.tif"

#     # Full paths
#     old_path = os.path.join(label_folder, filename)
#     new_path = os.path.join(label_folder, new_filename)

#     # Rename file
#     os.rename(old_path, new_path)
#     print(f"Renamed: {filename} --> {new_filename}")

# print("✅ Label files renamed successfully!")

## Run this to create train.txt and val.txt with 80/20 split from A folder filenames

# import os
# import random

# # Path to your dataset (use A folder since names are the same for B & labels)
# dataset_folder = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\A"

# # Output paths for train.txt and val.txt
# output_train = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\list/train.txt"
# output_val = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\list/val.txt"

# # Get all .tif file names
# all_files = sorted([f for f in os.listdir(dataset_folder) if f.lower().endswith(".tif")])

# # Shuffle for randomness
# random.seed(42)  # Fix seed for reproducibility
# random.shuffle(all_files)

# # 70/30 split ✅
# split_index = int(len(all_files) * 0.8)
# train_files = all_files[:split_index]
# val_files = all_files[split_index:]

# # Save train.txt with .tif filenames
# with open(output_train, "w") as f:
#     for name in train_files:
#         f.write(name + "\n")

# # Save val.txt with .tif filenames
# with open(output_val, "w") as f:
#     for name in val_files:
#         f.write(name + "\n")

# print(f"✅ Train/Val split created successfully!")
# print(f"Total files      : {len(all_files)}")
# print(f"Training files   : {len(train_files)}")
# print(f"Validation files : {len(val_files)}")
# print(f"train.txt saved at: {output_train}")
# print(f"val.txt saved at  : {output_val}")



