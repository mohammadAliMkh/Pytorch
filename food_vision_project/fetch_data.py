
import os
from pathlib import Path
import requests
from zipfile import ZipFile

def get_data(zip_file_name , raw_path_url):
  """
  Get zip file from url link

    args:
      zip_file_name: name of the zip file (include .zip)
      raw_path_url: url link that raw zip file (include .zip)
    
    return:
      train_dir: train direcotry path
      test_dir: test directory path
  """

  file_name = zip_file_name
  folder_name = file_name.split(".zip")[0]

  parent_dir = Path.cwd()
  data_dir = parent_dir/folder_name

  if Path.exists(data_dir):
    print(f"{folder_name} Folder Already Exists...")
  else:
    os.makedirs(data_dir)
    print(f"{folder_name} Folder Created...")



  # get pizza_steak_sushi.zip file from mrdbourrke github repository
  r = requests.get(raw_path_url)

  with open(file_name , "wb") as f:
    f.write(r.content)

  data_path = parent_dir/file_name
  print(f"{file_name} downloaded...")



  # loading the temp.zip and creating a zip object
  with ZipFile(data_path , 'r') as zObject:
    
      # Extracting all the members of the zip 
      # into a specific location.
      zObject.extractall(
          path=data_dir)
  print(f"{file_name} extracted at {data_dir}")
  # removing the downloaded.zip file from parent directory
  os.remove(data_path)

  train_dir = data_dir/"train"
  test_dir = data_dir/"test"

  return train_dir , test_dir
