#import multiprocessing
import time

from cv2 import cv2
from bs4 import BeautifulSoup
from xml.dom import minidom
import natsort
import numpy as np
import os
import json
import cv2
import requests
import logging
from multiprocessing.dummy import Pool as ThreadPool
# from torch.multiprocessing import Pool as ThreadPool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

from functools import partial

from paddleocr import PaddleOCR,draw_ocr
from pathlib import Path
from PIL import Image


def perform_ocr(image_path, ocr_engine=None):
    list_res = []
    results = ocr_engine.ocr(image_path, cls=False)

    for res in results:
        # list_res.append(res[1][0])
        [list_res.append(f) for f in res[1][0].split(" ")]
    print(f"image: {image_path}")
    print(f"{list_res}")
    print("--------------------------------------")
    return list_res

def ocr_space_file(filename, overlay=True, api_key='helloworld', language='eng'):
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
    stri = None
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

    try:
        r = r.content.decode()
        r = json.loads(r)
        parsed_results = r.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")
        # text_detected = text_detected.split('\r\n')
        text_detected = text_detected.split()
        stri = "\n".join(text_detected)
    except Exception as e:
        print(e)
    repr(stri)

    return stri


class PreProcess:

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True,lang='en')  # need to run only once to download and load model into memory
        self.kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])



    def xml_writing(self, images_path, path, id_xml):
        count = 0
        lst = natsort.natsorted(os.listdir(path), reverse=False)
        root = minidom.Document()
        xml = root.createElement('Drug')
        xml.attributes['id'] = str(id_xml)
        root.appendChild(xml)
        processes = os.cpu_count()

        # with multiprocessing.Pool(processes=processes) as pool:
        #     results = pool.map(partial(perform_ocr, ocr_engine=self.ocr), images_path)

        # pool = ThreadPool(processes)
        # #results = pool.map(ocr_space_file, images_path)
        # results = pool.map(self.perform_ocr, images_path)

        imread_arr = []
        for img in images_path:
            imread_arr.append(cv2.imread(img))

        pool = ThreadPool(processes)
        # #results = pool.map(ocr_space_file, images_path)
        results = pool.map(partial(perform_ocr, ocr_engine=self.ocr), imread_arr)

        #results = []
        for f in images_path:
            start = time.time()
            results.append(self.perform_ocr(f))
            stop = time.time()
            print("OCR time:" + str(stop - start))

        pool.close()
        pool.join()
        images_path.clear()
        for i, img in enumerate(lst):
            try:
                productChild = root.createElement('image')
                productChild.attributes['id'] = "%d" % count
                xml.appendChild(productChild)
                text = root.createTextNode(results[i])
                productChild.appendChild(text)
            except Exception as e:
                logging.warning(e)
            count += 1

        xml_str = root.toprettyxml(indent="\t")
        with open(path + '/strings_from_images.xml', "w", encoding='utf-8') as f:
            print("Save xml to" + path)
            f.write(xml_str)


    def reading_xml(self, path):
        value1 = 0
        value2 = 0
        try:
            path += '\\EQDrug.xml'
            print("reading_xml: " + path)
            # Reading the data inside the xml file to a variable under the name data
            with open(path, 'r') as f:
                data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            # Using find() to extract attributes
            b_name = Bs_data.find('Item')
            # Extracting the data stored in a specific attribute of the `VialHeight` tag
            value1 = b_name.get('VialHeight')
            value2 = b_name.get('ID')
        except Exception as e:
            logging.warning(e)

        return value1, value2


    def pre_process_proc(self, path):
        print("Inside Crop function")
        images_path = []
        vial_img = id_xml = ''
        PreProcessed = os.path.join(path, "PreProcessed")
        #ToTraining = os.path.join(path, "ToTraining")
        print("PathToCropped:"+PreProcessed)
        if not os.path.exists(PreProcessed):
            os.mkdir(PreProcessed)
        for f in os.listdir(path):
            if f.isnumeric():
                try:
                    count = 0
                    file_path = os.path.join(path, f)
                    h_xml, id_xml = self.reading_xml(file_path)
                    lst = os.listdir(file_path)
                    # x_min, x_max, y_min, y_max = detect.detect_from_an_image(file_path)
                    vial_img = os.path.join(PreProcessed, f)
                    if not os.path.exists(vial_img):
                        os.mkdir(vial_img)
                    for img in lst:
                        try:
                            if img.endswith(".png") or img.endswith(".bmp") or img.endswith(".jpg"):

                                img_array = cv2.imread(os.path.join(path, f, img))
                                # cropped_image = img_array[y_min:y_max, x_min:x_max]
                                cropped_image = cv2.filter2D(src=img_array, ddepth=-1, kernel=self.kernel)
                                cv2.imwrite(vial_img + '\\%d.jpg' % count, cropped_image)
                                images_path.append(vial_img + '\\%d.jpg' % count)
                        except Exception as e:
                            print("Exception: " + e)
                            logging.warning(e)
                        count += 1
                except Exception as e:
                    logging.warning(e)

                self.xml_writing(images_path, vial_img, id_xml)


if __name__ == '__main__':
    # path = r'C:\Users\itzik.k\Desktop\Vials'
    # pre = PreProcess()
    # pre.pre_process_proc(path)
    pre = PreProcess()
    path = r'C:\Users\itzik.k\Desktop\Vials - Paddle'
    #path = r'C:\Users\itzik.k\Desktop\Vials - Paddle\PreProcessed\1005'
    pre.pre_process_proc(path)



