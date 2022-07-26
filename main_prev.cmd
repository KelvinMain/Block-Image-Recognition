

cd python_scripts
python "Label and Split Dataset.py"
python create_annotations_and_create_label_map.py
python generate_tfrecord.py -x TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/train -l MODELS_DO_NOT_TOUCH/annotations/label_map.pbtxt -o MODELS_DO_NOT_TOUCH/annotations/train.record -i TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/photos -c MODELS_DO_NOT_TOUCH/annotations/train.csv
python generate_tfrecord.py -x TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/val -l MODELS_DO_NOT_TOUCH/annotations/label_map.pbtxt -o MODELS_DO_NOT_TOUCH/annotations/test.record -i TEMP_DO_NOT_TOUCH/getting_ready_to_create_record_file/photos -c MODELS_DO_NOT_TOUCH/annotations/test.csv
python create_new_ssd_to_be_trained_and_configure_pipeline_file.py
python model_main_tf2.py --model_dir=MODELS_DO_NOT_TOUCH/models/ssd15 --pipeline_config_path=MODELS_DO_NOT_TOUCH/models/ssd15/pipeline.config
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path MODELS_DO_NOT_TOUCH/models/ssd15/pipeline.config --trained_checkpoint_dir MODELS_DO_NOT_TOUCH/models/ssd15/ --output_directory MODELS_DO_NOT_TOUCH/exported-models/my_model_ssd15
cd ..


