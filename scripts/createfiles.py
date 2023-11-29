import os
import sys
import shutil

# input: dir/filename.md
# if dir not exist, create it
# create the filename.md with content "TODO"
def mkdir_touch_files(path):
  path = path.strip()
  dirname = path.split('/')[0]
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(path, 'w') as f:
    f.write('TODO')
  print('create file: ' + path)

files = []
for file in files:
  mkdir_touch_files(file)

  