import splitfolders
import os
import shutil


def split_dataset():
    # remove previous files at TEMP_DO_NOT_TOUCH/output (from last run)
    save_folder_path = 'TEMP_DO_NOT_TOUCH/output'
    for filename in os.listdir(save_folder_path):
        file_path = os.path.join(save_folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    splitfolders.ratio("TEMP_DO_NOT_TOUCH/xml_files", output="TEMP_DO_NOT_TOUCH/output", seed=1337, ratio=(.8, .2),
                       group_prefix=None)  # default values
