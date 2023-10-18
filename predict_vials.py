import os
import ast
import time
import torch
import cv2 as cv2
import random
from colorama import Fore
from colorama import Style
import numpy as np
from natsort import os_sorted
from paddleocr import PaddleOCR, draw_ocr
import json
import requests
import configparser
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.models import DenseNet121_Weights
from detect_label import LabelDetection
from multiprocessing.dummy import Pool as ThreadPool
from PaddleOCR import paddle_ocr


class PredictVial:

    def __init__(self):
        self.vials_map = {}
        self.val_string = []
        self.list = []
        self.keys = []
        self.config = self.reading_file_config('config_file.ini')
        self.label_model = LabelDetection()
        self.model = torch.load(os.path.join(self.config.get("Models", "model_vials"), 'best.pt'), map_location='cpu')
        self.model.eval()
        self.searchFlag_arr = [True, True, True]
        self.PredictExportFileItems = self.reading_xml()
        self.ocr = paddle_ocr()
        self.kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    def ocr_space_file(self, filename, overlay=False, api_key='helloworld', language='eng'):
        """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
        :param filename: Your file path & name.
        :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
        :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
        :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
        :return: Result in JSON format.
        """

        payload = {'isOverlayRequired': overlay,
                   'scale': True,
                   'apikey': api_key,
                   'language': language,
                   }
        with open(filename, 'rb') as f:
            r = requests.post('http://localhost:8081/parse/image',
                              files={filename: f},
                              data=payload,
                              )
            r = r.content.decode()
            r = json.loads(r)
            parsed_results = r.get("ParsedResults")[0]
            text_detected = parsed_results.get("ParsedText")
            text_detected = text_detected.split()

        return text_detected


    def getPredictExportFileItem(self):
        keys = self.vials_map.keys()

        for item in self.PredictExportFileItems:
            try:
                id = item.get('Id')
                if id in keys:
                    self.vials_map[id][1][0] = item.get('ValidationStringsGroup1')
                    self.vials_map[id][1][1] = item.get('ValidationStringsGroup2')
                    self.vials_map[id][1][2] = item.get('ValidationStringsGroup3')

                    if self.vials_map[id][1][0] != '':
                        valid_group1 = self.vials_map[id][1][0]
                        self.vials_map[id][1][0] = valid_group1.split("\r\n")
                        self.vials_map[id][1][0] = [i.strip(' ') for i in self.vials_map[id][1][0]]
                    # if validation string empty --> set flag to false in order to prevent searching
                    else:
                        self.vials_map[id][0][0] = False

                    if self.vials_map[id][1][1] != '':
                        valid_group2 = self.vials_map[id][1][1]
                        self.vials_map[id][1][1] = valid_group2.split("\r\n")
                        self.vials_map[id][1][1] = [i.strip(' ') for i in self.vials_map[id][1][1]]
                    else:
                        self.vials_map[id][0][1] = False

                    if self.vials_map[id][1][2] != '':
                        valid_group3 = self.vials_map[id][1][2]
                        self.vials_map[id][1][2] = valid_group3.split("\r\n")
                        self.vials_map[id][1][2] = [i.strip(' ') for i in self.vials_map[id][1][2]]
                    else:
                        self.vials_map[id][0][2] = False
            except Exception as e:
                print(e)

            # self.val_string.clear()
            # self.val_string.extend((val_string_1, val_string_2, val_string_3))

    def reading_xml(self):
        path = self.config.get("Paths", "validation_strings_info_file")
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            items = root.findall("Item")
            return items
        except Exception as e:
            print(e)

    def read_txt_file(self):
        txt_file = self.config.get("Paths", "text_file")
        txt_file = open(txt_file, "r+")
        for i, line in enumerate(txt_file):
            cls = line.split('\n')
            if cls[0] != '':
                self.keys.append(cls[0])

    def predict(self, image_path):
        score = 0
        prediction_name = None
        try:
            image = os.path.basename(image_path)
            weights = DenseNet121_Weights.DEFAULT
            preprocess = weights.transforms()

            # load image
            img = Image.open(image_path)

            # Transform
            input = preprocess(img)

            # Get prediction
            input = input.unsqueeze(0)
            prediction = self.model(input).squeeze(0).softmax(0)

            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            prediction_name = self.keys[class_id]

            print(f"In image {image} res is : {class_id} score is: {score}")

        except Exception as e:
            print(e)

        if score > 0.8:
            # high_score_prediction = [prediction_name, score, image_path[0]]
            return [prediction_name, score, image_path]
        else:
            return []

    def initVialsMap(self, prediction_results):
        for res in prediction_results:
            if res != [] and res[0] not in self.vials_map:
                self.vials_map[res[0]] = [True, True, True], ['', '', '']
        return self.vials_map

    def crop_coordinates(self, xyxy_tuple):
        x1, x2, y1, y2 = [], [], [], []
        for xyxy in xyxy_tuple:
            x1.append(xyxy[0]), y1.append(xyxy[1]), x2.append(xyxy[2]), y2.append(xyxy[3])
        return [min(x1), min(y1), max(x2), max(y2)]

    def crop_label(self, images_path, pool):
        path_list = []
        suffixes = (".png", ".bmp", ".jpg")
        images_dir = os.path.dirname(images_path[0])
        cropped_dir = os.path.join(images_dir, "cropped")
        labeled_dir = os.path.join(images_dir, "labeled")

        if not os.path.exists(cropped_dir):
            os.mkdir(cropped_dir)
        if not os.path.exists(labeled_dir):
            os.mkdir(labeled_dir)

        xyxy = pool.map(self.label_model.detect_label, images_path)
        xyxy = list(filter(None, xyxy))
        xyxy = self.crop_coordinates(xyxy)

        for image in os_sorted(os.listdir(images_dir)):
            try:
                if image.endswith(suffixes):
                    image_name = os.path.basename(image)
                    image_full_path = os.path.join(images_dir, image)

                    img = cv2.imread(image_full_path)

                    saving_path = os.path.join(cropped_dir, image_name)
                    # cropping and saving to  preparation directory
                    cropped_image = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    cv2.imwrite(saving_path, cropped_image)
                    path_list.append(saving_path)
            except Exception as e:
                print(e)
        return path_list

    def predict_procedure(self, images_path):
        self.read_txt_file()
        self.vials_map = {}
        processes = os.cpu_count()
        pool = ThreadPool(processes)
        start = time.time()

        suffixes = (".png", ".bmp", ".jpg")
        images_arr = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(suffixes)]

        random_images = random.sample(images_arr, 10)
        # xyxy = pool.map(self.label_model.detect_label, random_images)
        crop_image_path_list = self.crop_label(random_images, pool)
        stop = time.time()
        print("crop label time:" + str(stop - start))

        chunks = [crop_image_path_list[x:x + 10] for x in range(0, len(crop_image_path_list), 10)]
        for chunk in chunks:
            start_class_pred = time.time()
            result = pool.map(self.predict, chunk)
            result = list(filter(None, result))
            stop = time.time()
            print(f"predict time: {str(stop - start)}")
            print(f"predict classifier time: {str(stop - start_class_pred)}")
            self.vials_map = self.initVialsMap(result)

            for vial in result:
                if vial:
                    vial_name = os.path.basename(vial[2])
                    vial_dir = os.path.dirname(os.path.dirname(vial[2]))
                    vial_path = os.path.join(vial_dir, vial_name)

                    self.getPredictExportFileItem()
                    string_ocr = self.ocr.perform_ocr(vial_path)
                    print("")
                    print(f"Image path: {vial[2]}")
                    print(f"AI result: class={vial[0]}, score={vial[1]}")
                    print(f"Text (by OCR): {string_ocr}")

                    ocr_result = self.matching_strings(string_ocr)
                    if ocr_result != -1:
                        stop = time.time()
                        print(f"Total time: {stop - start}")
                        print(f"Vial predicted: {ocr_result}")
                        return ocr_result

        stop = time.time()
        print('can not find suitable vial')
        print(f"Total time: {stop - start}")
        return -1

    def matching_strings(self, string_ocr):
        ocr_txt_lower = [x.lower() for x in string_ocr]
        # if OCR NOT found words
        if not ocr_txt_lower:
            return -1
        keys = self.vials_map.keys()
        found = False
        for id in keys:
            found = False
            if self.vials_map[id][0][0]:
                validationGroup_lower = [x.lower() for x in self.vials_map[id][1][0]]
                for i in validationGroup_lower:
                    for word in ocr_txt_lower:
                        ## search for substring (only for validation group 1 - hold trade names)
                        if i in word:
                            self.vials_map[id][0][0] = False
                            found = True
                            # ocr_txt_lower.remove(i)
                            print(
                                f"{Fore.GREEN}{i} found in {word} - validation group 1 (vial ID: {id}){Style.RESET_ALL}")
                            break
                    if found:
                        break

            if self.vials_map[id][0][1]:
                validationGroup_lower = [x.lower() for x in self.vials_map[id][1][1]]
                for i in validationGroup_lower:
                    if i in ocr_txt_lower:
                        self.vials_map[id][0][1] = False
                        ocr_txt_lower.remove(i)
                        print(f"{Fore.GREEN}{i} found in validation group 2 (vial ID: {id}){Style.RESET_ALL}")
                        break

            if self.vials_map[id][0][2]:
                validationGroup_lower = [x.lower() for x in self.vials_map[id][1][2]]
                for i in validationGroup_lower:
                    for word in ocr_txt_lower:
                        ## search for substring (only for validation group 3 - hold trade names)
                        if i in word:
                            self.vials_map[id][0][2] = False
                            found = True
                            # ocr_txt_lower.remove(i)
                            print(
                                f"{Fore.GREEN}{i} found in {word} - validation group 3 (vial ID: {id}){Style.RESET_ALL}")
                            break
                    if found:
                        break

            if not self.vials_map[id][0][0] and not self.vials_map[id][0][1] \
                    and not self.vials_map[id][0][2]:
                return id
        return -1

    def reading_file_config(self, path_config):
        config_file = configparser.RawConfigParser()
        config_file.read(path_config)

        return config_file


if __name__ == "__main__":
    images_path = r'C:\Users\dor.s\Desktop\images\Lisbon\C03_18_06_2023_15_49_23_548'
    predict = PredictVial()
    result = predict.predict_procedure(images_path)
    print(result)
