from distutils.dir_util import copy_tree
import os
import shutil

# remove previous files at MODELS_DO_NOT_TOUCH/models/ssd15 (from last run)
if not os.path.exists('MODELS_DO_NOT_TOUCH/models/ssd15'):
    os.makedirs('MODELS_DO_NOT_TOUCH/models/ssd15')

save_folder_path = 'MODELS_DO_NOT_TOUCH/models/ssd15'
for filename in os.listdir(save_folder_path):
    file_path = os.path.join(save_folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Update pipeline.config
# change the number of classes, learning rate, batch size, label map path, input path to a suitable number and path.
batch_size = 1
while batch_size * 2 < len(os.listdir("TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/train")) and batch_size < 32:
    batch_size *= 2
number_of_classes = len(os.listdir("TEMP_DO_NOT_TOUCH/output/train"))
learning_rate = "IS NOT CHANGED FOR NOW"
# TODO - BETTER LEARNING RATE?
# TODO - AUTOMATE NUMBER OF STEP DECISION?
# TODO - maybe batch size of 8? 16? max
label_map_path = '\"MODELS_DO_NOT_TOUCH/annotations/label_map.pbtxt\"'
train_input_path = '\"MODELS_DO_NOT_TOUCH/annotations/train.record\"'
val_input_path = '\"MODELS_DO_NOT_TOUCH/annotations/test.record\"'
fine_tune_checkpoint = '\"MODELS_DO_NOT_TOUCH/pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0\"'

with open('MODELS_DO_NOT_TOUCH/pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config',
          'r') as reference:
    line = 1
    while line < 190:
        with open('MODELS_DO_NOT_TOUCH/models/ssd15/pipeline.config', 'a') as f:
            if line == 3:
                reference.readline()
                f.write('    num_classes: ' + str(number_of_classes) + '\n')
            elif line == 138:
                reference.readline()
                f.write('  batch_size: ' + str(batch_size) + '\n')
            elif line == 162:
                reference.readline()
                f.write('  fine_tune_checkpoint: ' + fine_tune_checkpoint + '\n')
            elif line == 172 or line == 182:
                reference.readline()
                f.write('  label_map_path: ' + label_map_path + '\n')
            elif line == 174:
                reference.readline()
                f.write('    input_path: ' + train_input_path + '\n')
            elif line == 186:
                reference.readline()
                f.write('    input_path: ' + val_input_path + '\n')
            else:
                f.write(reference.readline())
        line += 1
