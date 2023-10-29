import os

current_directory = "./Dataset/Dataset_Gen/labels_fog_60"  
current_folder = os.path.basename(current_directory)  
folder_paths = []  

for root, dirs, files in os.walk(current_directory):
    for dir in dirs:
        if root != current_directory:
            parent_folder = os.path.basename(root)
            folder_name = dir
            path = os.path.join(current_folder, parent_folder, folder_name)
            folder_paths.append(path)

with open("anno_paths.txt", "w") as file:
    for path in folder_paths:
        file.write(path + "\n")

