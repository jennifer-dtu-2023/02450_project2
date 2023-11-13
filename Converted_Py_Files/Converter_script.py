import os
import subprocess

root_dir = '/Users/jenniferfortuny/02450_project2'
destination_dir = '/Users/jenniferfortuny/02450_project2/Converted_Py_Files'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.ipynb'):
            file_path = os.path.join(subdir, file)
            subprocess.run(['jupyter', 'nbconvert', '--to', 'script', file_path])

            # Move the converted .py file
            base_name = os.path.splitext(file)[0]
            py_file = os.path.join(subdir, base_name + '.py')
            new_location = os.path.join(destination_dir, base_name + '.py')
            os.rename(py_file, new_location)
