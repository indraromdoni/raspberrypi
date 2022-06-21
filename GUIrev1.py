import tkinter
import os
import cv2 as cv
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = tkinter.Tk()
app.title("Lever Child Color Judge")
app.geometry("300x600")
masterlist = ["master00", "master01", "master02", "master03"]
masterselect = tkinter.StringVar(app)
masterselect.set("Select option")
cam = 0

#OK image
def takeimage(j):
    flag_ok = False
    cap = cv.VideoCapture(cam)
    while True:
        success, frame = cap.read()
        cv.imshow("Sample OK Image", frame)
        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite(j + "/master-image.jpg", frame)
            flag_ok = True
            break
        if key == 27:
            break
    if flag_ok == True:
        img = cv.imread(j + "/master-image.jpg")
        roi = cv.selectROI(img)
        roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        gray_cropped = cv.cvtColor(roi_cropped, cv.COLOR_BGR2GRAY)
        cv.imwrite(j + "/master-template.jpg", gray_cropped)
        cv.destroyAllWindows()
        i = 0
        while i<50:
            success, frame = cap.read()
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            template = cv.imread(j+"/master-template.jpg",0)
            w,h = template.shape[::-1]
            result = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            img_cropped = gray_img[top_left[1]:top_left[1] + h,top_left[0]:top_left[0] + w ]#img[top : bottom , left : right]
            strnum = str(i)
            cv.imwrite(j+"/image_data/img"+strnum+".jpg", img_cropped)
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
        lokasi = "master0"+i_str+"/image_data"
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
    with open("model.pickle", mode='wb') as f:
        pickle.dump(clf, f)
    with open("model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    preds = clf.predict(X_test)
    print("preds ", preds)
    print("Y_test ", Y_test)
    print("Accuracy Score ", accuracy_score(preds, Y_test, normalize=True, sample_weight=None))

def predict():
    predict = "Prediction start"
    print(predict)
    clf = GradientBoostingClassifier()
    with open("model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    X = []
    cap = cv.VideoCapture(cam)
    success, frame = cap.read()
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    template = cv.imread("master02/master-template.jpg", 0)
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
    print("Prediction : ", preds)
    modelselect = ["Lever Child White", "Lever Child Black", "No Lever Child", "No Lever Child 1"]
    cv.rectangle(frame, top_left, bottom_right, (0,255,0), thickness=2)
    cv.putText(frame, modelselect[preds[0]], org=top_left, fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
    cv.imshow("Result", frame)
    while True:
        if not cv.getWindowProperty("Result", cv.WND_PROP_VISIBLE):
            print("Operation Break")
            break
        key = cv.waitKey(50)
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()

def predictloop():
    predict = "Prediction start"
    print(predict)
    clf = GradientBoostingClassifier()
    with open("model.pickle", mode='rb') as f:
        clf = pickle.load(f)
    X = []
    cap = cv.VideoCapture(cam)
    while True:
        success, frame = cap.read()
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        template = cv.imread("master02/master-template.jpg", 0)
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
        if not cv.getWindowProperty("Result", cv.WND_PROP_VISIBLE):
            print("Operation Break")
            break
        key = cv.waitKey(30)
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()
        

def select(i):
    slc = "Selected master : "+i
    slcd = masterlist.index(masterselect.get())
    print(slcd)

mode = tkinter.IntVar(value=0)
def runmode():
    run = ["Running Mode", "Programming Mode"]
    print(run[mode.get()])
    if mode.get() == 0:
        part_00['state'] = tkinter.DISABLED
        part_01['state'] = tkinter.DISABLED
        part_02['state'] = tkinter.DISABLED
        part_03['state'] = tkinter.DISABLED
        part_04['state'] = tkinter.DISABLED
        part_05['state'] = tkinter.DISABLED
        part_051['state'] = tkinter.DISABLED
        part_052['state'] = tkinter.DISABLED
        part_053['state'] = tkinter.DISABLED
        part_054['state'] = tkinter.DISABLED
        part_055['state'] = tkinter.DISABLED
        part_06['state'] = tkinter.DISABLED
        part_07['state'] = tkinter.DISABLED
        part_08['state'] = tkinter.NORMAL
        part_09['state'] = tkinter.NORMAL
        part_10['state'] = tkinter.NORMAL
    else:
        part_00['state'] = tkinter.NORMAL
        part_01['state'] = tkinter.NORMAL
        part_02['state'] = tkinter.NORMAL
        part_03['state'] = tkinter.NORMAL
        part_04['state'] = tkinter.NORMAL
        part_05['state'] = tkinter.NORMAL
        part_051['state'] = tkinter.NORMAL
        part_052['state'] = tkinter.NORMAL
        part_053['state'] = tkinter.NORMAL
        part_054['state'] = tkinter.NORMAL
        part_055['state'] = tkinter.NORMAL
        part_06['state'] = tkinter.NORMAL
        part_07['state'] = tkinter.NORMAL
        part_08['state'] = tkinter.NORMAL
        part_09['state'] = tkinter.NORMAL
        part_10['state'] = tkinter.NORMAL

#widget list
part_00 = tkinter.Label(app, text="1. New Master")
part_01 = tkinter.OptionMenu(app, masterselect, *masterlist, command=takeimage, )
part_02 = tkinter.Button(app, text="Start Learning", command=learning)
part_03 = tkinter.Label(app, text="")
part_04 = tkinter.Label(app, text="2. Choose Master Program")
part_05 = tkinter.Label(app, text="Triger Input")
part_051 = tkinter.Entry(app, bd=2)
part_052 = tkinter.Label(app, text="Data Input")
part_053 = tkinter.Entry(app, bd=2)
part_054 = tkinter.Label(app, text="Judgement Result")
part_055 = tkinter.Entry(app, bd=2)
part_06 = tkinter.Button(app, text="Prediction Test", command=predict)
part_07 = tkinter.Label(app, text="")
part_08 = tkinter.Label(app, text="3. Choose Mode")
part_09 = tkinter.Radiobutton(app, text="Run", command=runmode, variable=mode, value=0)
part_10 = tkinter.Radiobutton(app, text="Program", command=runmode, variable=mode, value=1)
part_11 = tkinter.Button(app, text="Continues Trigger", command=predictloop)
#widget pack
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
part_06.pack()
part_07.pack()
part_08.pack()
part_09.pack()
part_10.pack()
part_11.pack()
runmode()
app.mainloop()