from paddleocr import PaddleOCR


class paddle_ocr:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='en')

    def perform_ocr(self, image_path):
        list_res = []
        results = self.ocr.ocr(image_path, cls=True)
        results = results[0]
        lines_list = [line[1][0] for line in results]
        list(map(lambda x: list_res.extend([list(x.split())][0]), lines_list))
        return lines_list