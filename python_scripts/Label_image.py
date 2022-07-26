"""
This script looks at all folders inside Images, for each folder, runs face detection to get face of block of all images
under it then create xml file created from this with the label being the same as the folder name
"""

import os
import shutil
import xml.etree.ElementTree as gfg

import cv2 as cv

import detect_edge_of_block as deob
import exceptions


def do_face_detection_and_labelling_for_all_img(TESTING):
    def do_face_detection_and_labelling_for_img(label_name, img, img_name):
        try:
            # face_detection
            [min_width, min_height, max_width, max_height] = deob.detect_side_of_block(img, testing_param=TESTING)
            final_image = cv.rectangle(img, (min_width, min_height), (max_width, max_height), (0, 255, 255), 5)
            cv.imwrite('TEMP_DO_NOT_TOUCH/photos with bounding box/' + img_name, final_image)

            if TESTING:
                cv.imshow(label_name, final_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            label = label_name

            # TODO - not allow empty label?
            # create_xml_file
            # search for label dir (if no, create)
            try:
                os.makedirs(save_folder_path + "/" + label)
            except FileExistsError:
                pass

            # new xml filename
            dimensions = img.shape
            height = dimensions[0]
            width = dimensions[1]
            depth = dimensions[2]
            xml_filename = img_name[:-3] + "xml"
            with open(save_folder_path + "/" + label + "/" + xml_filename, "x") as xml:
                xml.write("<annotation>\n\t<folder>"+label+"</folder>\n\t<filename>"+img_name+"</filename>"
                          "\n\t<path>D:/../Images/"+label+"/"+img_name+"</path>\n\t<source>\n\t\t<database>Unknown"
                          "</database>\n\t</source>\n\t<size>\n\t\t<width>"+str(width)+"</width>\n\t\t<height>"+str(height)+
                          "</height>\n\t\t<depth>"+str(depth)+"</depth>\n\t</size>\n\t<segmented>0</segmented>\n\t<object>"
                          "\n\t\t<name>"+label+"</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>"
                          "\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>"+str(min_width)+"</xmin>\n\t\t\t"
                          "<ymin>"+str(min_height)+"</ymin>\n\t\t\t<xmax>"+str(max_width)+"</xmax>\n\t\t\t<ymax>"+
                          str(max_height)+"</ymax>\n\t\t</bndbox>\n\t</object>\n</annotation>")

        except exceptions.NotEnoughLinesFoundOnImageError:
            print("image processing on image=" + img_name + " has failed, not enough lines "
                                                            "were found, image is skipped for now")
        except exceptions.FindLongestLineError:
            print("image processing on image=" + img_name + " has failed, lines found were not of the correct property "
                                                            ", image is skipped for now")
        except exceptions.ParallelLinesExpectedButNoneFoundError:
            print("image processing on image=" + img_name + " has failed, did not manage to find two pairs of parallel "
                                                            "lines, image is skipped for now")
        except exceptions.BoxFindingFailed:
            print(
                "image processing on image=" + img_name + " has failed, did not manage to find appropriate bounding "
                                                          "box , image is skipped for now")

    # remove previous files at TEMP_DO_NOT_TOUCH/xml_files (from last run)
    save_folder_path = 'TEMP_DO_NOT_TOUCH/xml_files'
    for filename in os.listdir(save_folder_path):
        file_path = os.path.join(save_folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for filename in os.listdir('TEMP_DO_NOT_TOUCH\photos with bounding box'):
        file_path = os.path.join('TEMP_DO_NOT_TOUCH\photos with bounding box', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    path = "../Images"
    for filename_ in os.listdir(path):
        for potential_photos in os.listdir(path + "/" + filename_):
            img_ = cv.imread(path + "/" + filename_ + "/" + potential_photos)
            if img_ is not None:
                do_face_detection_and_labelling_for_img(filename_, img_, potential_photos)
            else:
                print(filename_ + " is not an image, skipped")
