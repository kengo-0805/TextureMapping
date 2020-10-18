import cv2
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def arGenerator():
    for i in range(30):
        fileName = "/Users/horiikengo/PycharmProjects/Hellopython/data/{}.png".format(i)
        generator = aruco.drawMarker(dictionary, i, 100)
        cv2.imwrite(fileName, generator)
    cv2.waitKey(0)
arGenerator()