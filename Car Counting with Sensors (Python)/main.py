import cv2
import numpy as np

cap = cv2.VideoCapture("video2.mp4")
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
#5x5lik bir matris ve tüm değerler 1.  np.unit8=Unsigned integer (0 to 255)
kernel = np.ones((5,5), np.uint8)

class coordinat:
    def __init__(self, x, y):
        self.x = x
        self.y = y
#kullanım kolaylığı açısından sensor classı oluşturdum.
class sensor:
    def __init__(self, coordinat1, coordinat2, rectWidth, rectLength):
        self.coordinat1 = coordinat1
        self.coordinat2 = coordinat2
        self.rectWidth = rectWidth
        self.rectLength = rectLength
        self.maskArea = abs(self.coordinat2.x-coordinat1.x)*abs(self.coordinat2.y-self.coordinat1.y)
        #np.zeros 0'lardan oluşan bir dizin
        self.mask = np.zeros((rectLength, rectWidth, 1), np.uint8)
        cv2.rectangle(self.mask, (self.coordinat1.x, self.coordinat1.y), (self.coordinat2.x, self.coordinat2.y), (255), cv2.FILLED)
        self.state = False
        self.detectedVehicleNum = 0

sensor1 = sensor(coordinat(310,180), coordinat(420,240), 1080,250)
sensor2 = sensor(coordinat(150,180), coordinat(250,240), 1080,250)
sensor3 = sensor(coordinat(700,30), coordinat(810,90), 1080,250)
sensor4 = sensor(coordinat(580,30), coordinat(670,90), 1080,250)



while (1):
    ret, rect = cap.read()

    cutedRect = rect[350:600,100:1180]

    deletedBackGround = subtractor.apply(cutedRect)
    deletedBackGround = cv2.morphologyEx(deletedBackGround, cv2.MORPH_OPEN, kernel)
    ret, deletedBackGround = cv2.threshold(deletedBackGround, 127, 255, cv2.THRESH_BINARY)


    cnts, hierarchy= cv2.findContours(deletedBackGround, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #kesilmiş halinin kopyasını aldım.
    result = cutedRect.copy()
    #sensorler olacağı için oluşturdum.
    filledImage = np.zeros((cutedRect.shape[0], cutedRect.shape[1],1), np.uint8)



    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        #küçük kareleri almaması için koşulu koydum.
        if (w>30 and h>30):
            cv2.rectangle(result, (x,y),(x+w,y+h),(0,255,0),4 )
            cv2.rectangle(filledImage, (x,y), (x+w,y+h), (255), cv2.FILLED)



    sensor1MaskResult = cv2.bitwise_and(filledImage, filledImage,mask = sensor1.mask)
    #maskeye giren beyaz piksel sayısıyla maskenin boyutunu oranladım.daha sonrasında bu oranlarla işlem yapılacak.
    sensor1WhitePixelNum = np.sum(sensor1MaskResult==255)
    #kullanılan maskArea sensor classından geliyor.
    sensor1Rate = sensor1WhitePixelNum / sensor1.maskArea

    sensor2MaskResult = cv2.bitwise_and(filledImage, filledImage, mask=sensor2.mask)
    sensor2WhitePixelNum = np.sum(sensor2MaskResult == 255)
    sensor2Rate = sensor2WhitePixelNum / sensor2.maskArea

    sensor3MaskResult = cv2.bitwise_and(filledImage, filledImage, mask=sensor3.mask)
    sensor3WhitePixelNum = np.sum(sensor3MaskResult == 255)
    sensor3Rate = sensor3WhitePixelNum / sensor3.maskArea

    sensor4MaskResult = cv2.bitwise_and(filledImage, filledImage, mask=sensor4.mask)
    sensor4WhitePixelNum = np.sum(sensor4MaskResult == 255)
    sensor4Rate = sensor4WhitePixelNum / sensor4.maskArea

    #geçiş esnasında algılamak için state false--->true'ya geçiyor
    if (sensor1Rate >= 0.75 and sensor1.state == False):
        cv2.rectangle(result, (sensor1.coordinat1.x, sensor1.coordinat1.y),(sensor1.coordinat2.x, sensor1.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor1.state = True
    elif (sensor1Rate <= 0.75 and sensor1.state == True):
        cv2.rectangle(result, (sensor1.coordinat1.x, sensor1.coordinat1.y),(sensor1.coordinat2.x, sensor1.coordinat2.y), (0, 0, 255), cv2.FILLED)
        sensor1.state = False
        sensor1.detectedVehicleNum += 1
    else:
        cv2.rectangle(result, (sensor1.coordinat1.x, sensor1.coordinat1.y),(sensor1.coordinat2.x, sensor1.coordinat2.y), (0, 0, 255), cv2.FILLED)

    if (sensor2Rate >= 0.85 and sensor2.state == False):
        cv2.rectangle(result, (sensor2.coordinat1.x, sensor2.coordinat1.y),(sensor2.coordinat2.x, sensor2.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor2.state = True
    elif (sensor2Rate <= 0.85 and sensor2.state == True):
        cv2.rectangle(result, (sensor2.coordinat1.x, sensor2.coordinat1.y),(sensor2.coordinat2.x, sensor2.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor2.state = False
        sensor2.detectedVehicleNum += 1
    else:
        cv2.rectangle(result, (sensor2.coordinat1.x, sensor2.coordinat1.y),(sensor2.coordinat2.x, sensor2.coordinat2.y), (0, 0, 255), cv2.FILLED)

    if (sensor3Rate >= 0.10 and sensor3.state == False):
        cv2.rectangle(result, (sensor3.coordinat1.x, sensor3.coordinat1.y),(sensor3.coordinat2.x, sensor3.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor3.state = True
    elif (sensor3Rate <= 0.10 and sensor3.state == True):
        cv2.rectangle(result, (sensor3.coordinat1.x, sensor3.coordinat1.y),(sensor3.coordinat2.x, sensor3.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor3.state = False
        sensor3.detectedVehicleNum += 1
    else:
        cv2.rectangle(result, (sensor3.coordinat1.x, sensor3.coordinat1.y),(sensor3.coordinat2.x, sensor3.coordinat2.y), (0, 0, 255), cv2.FILLED)

    if (sensor4Rate >= 0.10 and sensor4.state == False):
        cv2.rectangle(result, (sensor4.coordinat1.x, sensor4.coordinat1.y),(sensor4.coordinat2.x, sensor4.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor4.state = True
    elif (sensor4Rate <= 0.10 and sensor4.state == True):
        cv2.rectangle(result, (sensor4.coordinat1.x, sensor4.coordinat1.y),(sensor4.coordinat2.x, sensor4.coordinat2.y), (0, 255, 0), cv2.FILLED)
        sensor4.state = False
        sensor4.detectedVehicleNum += 1
    else:
        cv2.rectangle(result, (sensor4.coordinat1.x, sensor4.coordinat1.y),(sensor4.coordinat2.x, sensor4.coordinat2.y), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_ITALIC

    cv2.putText(result, str(sensor1.detectedVehicleNum), (sensor1.coordinat1.x, sensor1.coordinat1.y + 60), font, 3, (255,255,255))
    cv2.putText(result, str(sensor2.detectedVehicleNum), (sensor2.coordinat1.x, sensor2.coordinat1.y + 60), font,3, (255, 255, 255))
    cv2.putText(result, str(sensor3.detectedVehicleNum), (sensor3.coordinat1.x, sensor3.coordinat1.y + 60), font, 3, (255, 255, 255))
    cv2.putText(result, str(sensor4.detectedVehicleNum), (sensor4.coordinat1.x, sensor4.coordinat1.y + 60), font, 3,(255, 255, 255))





    #cv2.imshow("Rect",rect)
    #cv2.imshow("Cuted Rect",cutedRect)
    #cv2.imshow("Deleted Back Ground", deletedBackGround)
    #cv2.imshow("Filled image", filledImage)
    #cv2.imshow("Sensor1 Mask Result", sensor4MaskResult)
    cv2.imshow("result",result)


    k= cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




