import cv2
import numpy as np

aruco = cv2.aruco
# マーカーの辞書選択
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)
# マーカーのサイズ
marker_length = 0.056 # [m]

camera_matrix = np.array([[322.742, 0, 617.92],
                        [ 0, 618.143, 237.32],
                        [ 0, 0, 1]])
distortion_coeff =  np.array(([0,0,0,0,0]))



while True:
    ret, img = capture.read()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
    # 可視化
    aruco.drawDetectedMarkers(img, corners, ids) 
    # resize the window
    windowsize = (800, 600)
    img = cv2.resize(img, windowsize)
    

    cv2.imshow('title',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(corners) > 0:
        # マーカーごとに処理
        for i, corner in enumerate(corners):
            # rvec -> rotation vector, tvec -> translation vector
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)
            # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = cv2.Rodrigues(rvec)
            rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
            print("id ={}".format(ids[i]))
            print(tvec)

capture.release()
cv2.destroyAllWindows()
