import os
import shutil

# clear annotations
if not os.path.exists('MODELS_DO_NOT_TOUCH/annotations'):
    os.makedirs('MODELS_DO_NOT_TOUCH/annotations')

# remove previous files at TEMP_DO_NOT_TOUCH/xml_files (from last run)
save_folder_path = 'MODELS_DO_NOT_TOUCH/annotations'
for filename in os.listdir(save_folder_path):
    file_path = os.path.join(save_folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# writing label_map into annotations
count = 1
for filename in os.listdir('TEMP_DO_NOT_TOUCH/xml_files'):
    with open('MODELS_DO_NOT_TOUCH/annotations/label_map.pbtxt', 'a') as f:
        f.write("item {\n  name: \"" + filename + "\"\n  id: " + str(count) + "\n}\n\n")
        count += 1
