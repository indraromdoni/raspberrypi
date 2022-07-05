import csv
from datetime import datetime
from subprocess import check_output
import time
import socket
import os
import cv2 as cv
import numpy as np
import pickle
import json
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk

root = Tk()
root.title('Lever Child Detection')
#root.iconbitmap('/home/pi/lever_child_project/')
root.geometry("400x600+100+50")
masterlist = ["master00", "master01", "master02", "master03"]
masterselect = StringVar(root)
masterselect.set("Select option")
triggerInput = StringVar(root)
dataInput = StringVar(root)
judgeOutput = StringVar(root)
ipaddress = StringVar(root)
myip = StringVar(root)
cam = 0
parentfld = '/home/pi/lever_child_project/'

#OK image
def takeimage(j):
    flag_ok = False
    cap = cv.VideoCapture(cam)
    f = open(parentfld+'default.json', 'r')
    data = json.load(f)
    f.close()
    _x, _y, _w, _h = data['roi']
    while True:
        success, frame = cap.read()
        if j[-1] != "0":
            cv.rectangle(frame, (_x, _y), (_x+_w, _y+_h), (0,255,0), 2)
        cv.imshow("Sample OK Image", frame)
        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite(parentfld + j + "/master-image.jpg", frame)
            flag_ok = True
            break
        if key == 27:
            break
    if flag_ok == True:
        img = cv.imread(parentfld + j + "/master-image.jpg")
        roi_cropped = 0
        if j[-1] == "0":
            f = open("default.json", "w")
            roi = cv.selectROI(img)
            data['roi'] = roi
            json.dump(data, f)
            f.close()
            print(data['roi'])
            roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        else:
            roi_cropped = img[_y:_y+_h, _x:_x+_w]
        gray_cropped = cv.cvtColor(roi_cropped, cv.COLOR_BGR2GRAY)
        cv.imwrite(parentfld + j + "/master-template.jpg", gray_cropped)
        cv.destroyAllWindows()
        i = 0
        while i<50:
            success, frame = cap.read()
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            template = cv.imread(parentfld+j+"/master-template.jpg",0)
            w,h = template.shape[::-1]
            result = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            img_cropped = gray_img[top_left[1]:top_left[1] + h,top_left[0]:top_left[0] + w ]#img[top : bottom , left : right]
            strnum = str(i)
            cv.imwrite(parentfld+j+"/image_data/img"+strnum+".jpg", img_cropped)
            print("Image data captured "+strnum)
            i = i + 1
        #takeimageNG(j=j)
    cv.destroyAllWindows()

def learning():
    start = "Learning start"
    print(start)
    img_size = 64
    X = []
    Y = []
    i = 0
    while i<4:
        i_str = str(i)
        lokasi = parentfld+"master0"+i_str+"/image_data"
        for base, dirs, files in os.walk(lokasi):
            print("Searching in :", base)
            cnt = 0
            for file in files:
                if cnt == 50:
                    break
                cntstr = str(cnt)
                img = cv.imread(lokasi+"/img"+cntstr+".jpg", 0)
                img_resize = cv.resize(img,(img_size, img_size))
                data = np.asarray(img_resize)
                data = data.reshape([-1])
                print(np.shape(data))
                X.append(data)
                Y.append(i)
                cnt += 1
        i+=1
        print(i) 
    print(np.shape(X))
    print(np.shape(Y))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, Y_train)
    with open(parentfld+"model.pickle", mode='wb') as f:
        pickle.dump(clf, f)
    with open(parentfld+"model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    preds = clf.predict(X_test)
    print("preds ", preds)
    print("Y_test ", Y_test)
    print("Accuracy Score ", accuracy_score(preds, Y_test, normalize=True, sample_weight=None))

f = open(parentfld+'default.json', 'r')
data = json.load(f)
print(data)
f.close()
# Create a client socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
# Connect to the server
try:
    clientSocket.connect((data['ipaddress'],8501));
except:
    print('Communication error to '+data['ipaddress'])

def predict():
    predict = "Prediction start"
    print(predict)
    cv.namedWindow('Result')
    cv.moveWindow('Result', 550, 50)
    clf = GradientBoostingClassifier()
    with open(parentfld+"model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    X = []
    cap = cv.VideoCapture(cam)
    success, frame = cap.read()
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    template = cv.imread(parentfld+"master00/master-template.jpg", 0)
    w,h = template.shape[::-1]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    print(res)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_cropped = img_gray[top_left[1]:top_left[1] + h,top_left[0]:top_left[0] + w ]#img[top : bottom , left : right]
    img_resize = cv.resize(img_cropped, (64, 64))
    data = np.asarray(img_resize)
    data = data.reshape([-1])
    X.append(data)
    preds = clf.predict(X)
    modelselect = ["Lever Child White", "Lever Child Black", "No Lever Child", "No Lever Child 1"]
    dataread = "RD "+dataInput.get()+"\r"
    ng = 0
    try:
        clientSocket.send(dataread.encode())
        msg = clientSocket.recv(1024)
        baca = msg.decode()
        baca = baca.replace('\n','')
        baca = baca.replace('\r','')
        int_baca = int(baca)
        cv.putText(frame, modelselect[int_baca], org=(5, 20), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 191, 255), thickness=1)
        print(int_baca)
        if preds[0] == int_baca:
            print("Good Product")
            cv.imwrite(parentfld+'result/'+str(datetime.now())+'.jpg', frame)
            data = "WR "+judgeOutput.get()+" 1\r"
            print(data)
            clientSocket.send(data.encode())
            psn = clientSocket.recv(1024)
            psn = psn.decode()
            print(psn)
            time.sleep(1)
            data1 = "WR "+judgeOutput.get()+" 0\r"
            print(data1)
            clientSocket.send(data1.encode())
            psn1 = clientSocket.recv(1024)
            psn1 = psn1.decode()
            print(psn1)
        else:
            cv.imwrite(parentfld+'resultng/'+str(datetime.now())+'.jpg', frame)
            print("NG Product, the product is "+modelselect[preds[0]])
            ng = 1
    except:
        print('Communication error')
    print("Prediction : ", preds)
    ng_str = ['OK', 'NG']
    logging_result = [datetime.now(), modelselect[int_baca], modelselect[preds[0]], ng_str[ng]]
    with open(parentfld+'logging_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(logging_result)
    ng = 0
    cv.rectangle(frame, top_left, bottom_right, (0,255,0), thickness=2)
    cv.putText(frame, modelselect[preds[0]], org=top_left, fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
    if mode.get() == 0:
        cv.imwrite(parentfld+'result/logging_result.jpg', frame)
        cv.imshow("Result", frame)
        cv.waitKey(200)
        cap.release()
        #cv.destroyAllWindows()
        runmode()
    else:
        cv.imshow("Result", frame)
        print(cv.getWindowProperty("Result", 0))
        while True:
            try:
                if cv.getWindowProperty("Result", 0)>0:
                    print(cv.getWindowProperty("Result", 0))
                    print("Operation Break")
                    break
            except:
                print("Window closed")
                break
            key = cv.waitKey(50)
            if key == 27:
                break
        cap.release()
        #cv.destroyAllWindows()

def predictloop():
    predict = "Prediction start"
    print(predict)
    clf = GradientBoostingClassifier()
    with open(parentfld+"model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    X = []
    cap = cv.VideoCapture(cam)
    cv.namedWindow('Result')
    cv.moveWindow('Result', 550, 50)
    while True:
        success, frame = cap.read()
        if not success:
            print('Cannot take picture')
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        template = cv.imread(parentfld+"master00/master-template.jpg", 0)
        w,h = template.shape[::-1]
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        #print(res)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img_cropped = img_gray[top_left[1]:top_left[1] + h,top_left[0]:top_left[0] + w ]#img[top : bottom , left : right]
        img_resize = cv.resize(img_cropped, (64, 64))
        data = np.asarray(img_resize)
        data = data.reshape([-1])
        X.append(data)
        preds = clf.predict(X)
        print("Prediction : ", preds)
        modelselect = ["Lever Child White", "Lever Child Black", "No Lever Child", "No Lever Child 1"]
        cv.rectangle(frame, top_left, bottom_right, (0,255,0), thickness=2)
        cv.putText(frame, modelselect[preds[-1]], org=top_left, fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
        cv.imshow("Result", frame)
        X = []
        if not cv.getWindowProperty("Result", cv.WND_PROP_VISIBLE):
            print("Window closed")
            break
        key = cv.waitKey(30)
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()
        
def setdefault():
    ip = check_output(['hostname', '-I'])
    myip.set(ip)
    f = open(parentfld+'default.json', 'r')
    data = json.load(f)
    print(data)
    f.close()
    print("ROI x : "+str(data['roi'][0]))
    print("ROI y : "+str(data['roi'][1]))
    print("ROI w : "+str(data['roi'][2]))
    print("ROI h : "+str(data['roi'][3]))
    triggerInput.set(data['triggerInput'])
    dataInput.set(data['dataInput'])
    judgeOutput.set(data['judgeOutput'])
    ipaddress.set(data['ipaddress'])
    if mode.get() == 0:
        runmode()
    
def wrdefault():
    f = open(parentfld+'default.json', 'r')
    data = json.load(f)
    print(data)
    f.close()
    f = open(parentfld+'default.json', 'w')
    data['triggerInput'] = triggerInput.get()
    data['dataInput'] = dataInput.get()
    data['judgeOutput'] = judgeOutput.get()
    data['ipaddress'] = ipaddress.get()
    json.dump(data, f)
    f.close()
    setdefault()

def select(i):
    slc = "Selected master : "+i
    slcd = masterlist.index(masterselect.get())
    print(slcd)

mode = IntVar(value=0)
def runmode():
    run = ["Running Mode", "Programming Mode"]
    print(run[mode.get()])
    if mode.get() == 0:
        part_00['state'] = DISABLED
        part_01['state'] = DISABLED
        part_02['state'] = DISABLED
        part_03['state'] = DISABLED
        part_04['state'] = DISABLED
        part_05['state'] = DISABLED
        part_051['state'] = DISABLED
        part_052['state'] = DISABLED
        part_053['state'] = DISABLED
        part_054['state'] = DISABLED
        part_055['state'] = DISABLED
        part_056['state'] = DISABLED
        part_057['state'] = DISABLED
        part_058['state'] = DISABLED
        part_059['state'] = DISABLED
        part_06['state'] = DISABLED
        part_061['state'] = DISABLED
        part_07['state'] = DISABLED
        part_08['state'] = NORMAL
        part_09['state'] = NORMAL
        part_10['state'] = NORMAL
        print("Run")
        dataread = "RD "+triggerInput.get()+"\r"
        try:
            clientSocket.send(dataread.encode())
            msg = clientSocket.recv(1024)
            rep = msg.decode()
            print(rep[0])
            if rep[0] == '1':
                print("Prediction Start")
                root.after(50, predict)
            else:
                root.after(50, runmode)
        except:
            print('Communication error')
            print('Reconnecting to ' + ipaddress.get())
            # Connect to the server
            try:
                clientSocket.connect((ipaddress.get(),8501))
                root.after(1000, runmode)
            except:
                root.after(1000, runmode)
    else:
        part_00['state'] = NORMAL
        part_01['state'] = NORMAL
        part_02['state'] = NORMAL
        part_03['state'] = NORMAL
        part_04['state'] = NORMAL
        part_05['state'] = NORMAL
        part_051['state'] = NORMAL
        part_052['state'] = NORMAL
        part_053['state'] = NORMAL
        part_054['state'] = NORMAL
        part_055['state'] = NORMAL
        part_056['state'] = NORMAL
        part_057['state'] = NORMAL
        part_058['state'] = NORMAL
        part_059['state'] = NORMAL
        part_06['state'] = NORMAL
        part_061['state'] = NORMAL
        part_07['state'] = NORMAL
        part_08['state'] = NORMAL
        part_09['state'] = NORMAL
        part_10['state'] = NORMAL

# Create A Main Frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)

# Create A Canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add A Scrollbar To The Canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)

# Configure The Canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

# Create ANOTHER Frame INSIDE the Canvas
second_frame = Frame(my_canvas)

# Add that New frame To a Window In The Canvas
my_canvas.create_window((0,0), window=second_frame, anchor="nw")

#widget list
part_0 = Label(second_frame, text="0. My Ip Address")
part_1 = Label(second_frame, text="192.168.0.1", textvariable=myip)
part_00 = Label(second_frame, text="1. Set Master")
part_01 = OptionMenu(second_frame, masterselect, *masterlist, command=takeimage, )
part_02 = Button(second_frame, text="Start Learning", command=learning)
part_03 = Label(second_frame, text="")
part_04 = Label(second_frame, text="2. General Setting")
part_05 = Label(second_frame, text="Triger Input Ex. MR0")
part_051 = Entry(second_frame, bd=2, textvariable=triggerInput)
part_052 = Label(second_frame, text="Data Input Ex. DM1")
part_053 = Entry(second_frame, bd=2, textvariable=dataInput)
part_054 = Label(second_frame, text="Judgement Result Ex. MR1")
part_055 = Entry(second_frame, bd=2, textvariable=judgeOutput)
part_056 = Label(second_frame, text="PLC IP Address (need to restart)")
part_057 = Entry(second_frame, bd=2, textvariable=ipaddress)
part_058 = Button(second_frame, text="Set Default", command=wrdefault)
part_059 = Label(second_frame, text="")
part_06 = Label(second_frame, text="3. Manual Trigger")
part_061 = Button(second_frame, text="Trigger", command=predict)
part_07 = Label(second_frame, text="")
part_08 = Label(second_frame, text="4. Choose Mode")
part_09 = Radiobutton(second_frame, text="Run", command=runmode, variable=mode, value=0)
part_10 = Radiobutton(second_frame, text="Program", command=runmode, variable=mode, value=1)
part_11 = Button(second_frame, text="Continues Trigger", command=predictloop)
#widget pack
part_0.pack()
part_1.pack()
part_00.pack()
part_01.pack()
part_02.pack()
part_03.pack()
part_04.pack()
part_05.pack()
part_051.pack()
part_052.pack()
part_053.pack()
part_054.pack()
part_055.pack()
part_056.pack()
part_057.pack()
part_058.pack()
part_059.pack()
part_06.pack()
part_061.pack()
part_07.pack()
part_08.pack()
part_09.pack()
part_10.pack()
part_11.pack()
setdefault()
root.mainloop()
