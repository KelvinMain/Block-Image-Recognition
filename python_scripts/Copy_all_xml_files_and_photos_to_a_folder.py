import shutil
import os


def copying():
    for filename in os.listdir('TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file'):
        file_path = os.path.join('TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    try:
        os.makedirs('TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/train')
        os.makedirs('TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/val')
        os.makedirs('TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/photos')
    except FileExistsError:
        pass
    # Copy (train)
    for folder in os.listdir('TEMP_DO_NOT_TOUCH/output/train'):
        for xml_file in os.listdir('TEMP_DO_NOT_TOUCH/output/train/' + folder):
            shutil.copy('TEMP_DO_NOT_TOUCH/output/train/' + folder + '/' + xml_file,
                        'TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/train/' + xml_file)

    # Copy (val)
    for folder in os.listdir('TEMP_DO_NOT_TOUCH/output/val'):
        for xml_file in os.listdir('TEMP_DO_NOT_TOUCH/output/val/' + folder):
            shutil.copy('TEMP_DO_NOT_TOUCH/output/val/' + folder + '/' + xml_file,
                        'TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/val/' + xml_file)

    # Copy (Photos)
    for folder in os.listdir('../Images'):
        for photos in os.listdir('../Images/' + folder):
            if photos[-4:] == ".jpg":
                shutil.copy('../Images/' + folder + '/' + photos,
                            'TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/photos/' + photos)
