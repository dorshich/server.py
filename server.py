import os
import socket
import threading
from pre_process import PreProcess
from train import Training
from predict_vials import PredictVial
from detect_label import LabelDetection
#import labeling.labelImg as labeling

message=[]
HEADER = 1024
PORT = 2001
hostName =  socket.gethostname()
#SERVER = socket.gethostbyname(socket.gethostname())
SERVER = '127.0.0.1'
ADDR = (SERVER, PORT)
DISCONNECT_MESSAGE = "DISCONNECT!"
PRE_PROCESS = 'start pre-process'
TRAINING = 'start training'
PREDICT = 'start predict-labels'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

pre_process = None
training = None
predict_vial = None
detect_label = None
path_config = None
path_logger = None
detect = None


def handle_client(conn, addr):
    global pre_process
    global training
    global predict_vial
    global detect_label
    global path_config
    global path_logger
    global detect

    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg = conn.recv(HEADER)
        if msg:
            if msg != DISCONNECT_MESSAGE:
                try:
                    paramsArr = str(msg, 'UTF-8').split("|")
                    print('Got command:' + paramsArr[0])
                    if paramsArr[0] == 'init_classes':
                        pre_process = PreProcess()
                        #training = Training(path_config, path_logger)
                        res = "init completed"
                    elif paramsArr[0] == 'PreProcess':
                        pre_process.pre_process_proc(paramsArr[1])
                        res = "PreProcess completed"
                    elif paramsArr[0] == 'labeling':
                        #os.system('python labeling/labelImg.py')
                        imagesPath = paramsArr[1]
                        if (detect_label == None):
                            labelDetect = LabelDetection()
                        labelDetect.pre_training(imagesPath)
                        res = "labeling completed"
                    elif paramsArr[0] == 'train':
                        if (training == None):
                            training = Training(paramsArr[1])
                        training.training(paramsArr[1])
                        res = "train completed"
                    elif paramsArr[0] == 'predict':
                        if(predict_vial == None):
                            predict_vial = PredictVial()
                        predicationRes = predict_vial.predict_procedure(paramsArr[1])
                        res = "predict completed|"+(str)(predicationRes)

                except Exception as e:
                    print(e)
                    res = e
            else:
                connected = False

            print(res)
            conn.send(res.encode())

    conn.close()


def start():
    server.listen()
    print(f"Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print("Establish connection with client")


if __name__ == '__main__':
    print("server is starting...")
    start()