import os
import ast
import time
from cv2 import cv2
import numpy
import numpy as np
import logging
import json
import requests
import configparser
from os import listdir
import xml.etree.ElementTree as ET
from detect_vial import VialDetection
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

class PredictVial:

    def __init__(self):
        self.config = self.reading_file_config('config_file.ini')
        self.vials_model = VialDetection()
        self.val_string = []
        self.list = []
        self.searchFlag_arr = [True, True, True]
        self.PredictExportFileItems = self.reading_xml()
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


    def getPredictExportFileItem(self, itemID):
        val_string_1 = None
        val_string_2 = None
        val_string_3 = None

        for item in self.PredictExportFileItems:
            id = item.get('Id')
            if id == itemID:
                val_string_1 = item.get('ValidationStringsGroup1')
                val_string_2 = item.get('ValidationStringsGroup2')
                val_string_3 = item.get('ValidationStringsGroup3')
                break
        if val_string_1 is not None:
            val_string_1 = val_string_1.split("\r\n")
            val_string_1 = [i.strip(' ') for i in val_string_1]
        # if validation string empty --> set flag to false in order to prevent searching
        else:
            self.searchFlag_arr[0] = False

        if val_string_2 != '':
            val_string_2 = val_string_2.split("\r\n")
            val_string_2 = [i.strip(' ') for i in val_string_2]
        else:
            self.searchFlag_arr[1] = False

        if val_string_3 != '':
            val_string_3 = val_string_3.split("\r\n")
            val_string_3 = [i.strip(' ') for i in val_string_3]
        else:
            self.searchFlag_arr[2] = False

        self.val_string.clear()
        self.val_string.extend((val_string_1, val_string_2, val_string_3))


    def reading_xml(self):
        path = self.config.get("Paths", "validation_strings_info_file")
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            items = root.findall("Item")
            return items
        except Exception as e:
            logging.warning(e)


    def predict(self, images_path):
        self.searchFlag_arr = [True, True, True]

        # create "cropped" dir before multiprocess in order to prevent conflict
        cropped_dir = os.path.join(images_path, "cropped")
        if not os.path.exists(cropped_dir):
            os.mkdir(cropped_dir)

        start = time.time()

        suffixes = (".png", ".bmp", ".jpg")
        images_arr = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(suffixes)]
        processes = os.cpu_count()
        pool = ThreadPool(processes)
        result = pool.map(self.vials_model.detect_vial, images_arr)

        stop = time.time()
        print("predict time:" + str(stop - start))

        start = time.time()
        for vial in result:
            if vial != []:
                self.getPredictExportFileItem(vial[0][0])
                string_ocr = self.ocr_space_file(filename=vial[0][2], language='eng')
                ocr_result = self.matching_strings(string_ocr)
                if ocr_result:
                    stop = time.time()
                    print("image of predict = " + vial[0][2])
                    print("Vial predicted: ", vial[0][0])
                    print("OCR time:" + str(stop - start))
                    return vial[0][0]

        print('can not find suitable vial')
        return None


    def matching_strings(self, string_ocr):
        if self.searchFlag_arr[0]:
            for i in self.val_string[0]:
                if i in string_ocr:
                    self.searchFlag_arr[0] = False
                    string_ocr.remove(i)
                    logging.warning(i + ' remove from ocr list')
                    break

        if self.searchFlag_arr[1]:
            for i in self.val_string[1]:
                if i in string_ocr:
                    self.searchFlag_arr[1] = False
                    string_ocr.remove(i)
                    logging.warning(i + ' remove from ocr list')
                    break

        if self.searchFlag_arr[2]:
            for i in self.val_string[2]:
                if i in string_ocr:
                    self.searchFlag_arr[2] = False
                    break

        if not self.searchFlag_arr[0] and not self.searchFlag_arr[1] and not self.searchFlag_arr[2]:
            return True
        else:
            return False


    def reading_file_config(self, path_config):
        config_file = configparser.RawConfigParser()
        config_file.read(path_config)

        return config_file


if __name__ == "__main__":
    images_path = r'C:\Users\itzik.k\Desktop\Vials\Test\1006'
    predict = PredictVial()
    result = predict.predict(images_path)
