import cv2
import numpy as np

aruco = cv2.aruco
cap = cv2.VideoCapture(0)

## C920の最大解像度を指定する width = 2304 height = 1536 fps = 2
cap.set(cv2.CAP_PROP_FPS, 2) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)

# 実際に利用できている解像度を取得する
ret, frame = cap.read() 
height, width, channels = frame.shape[:3]
print("width:{} hwight:{}".format(width, height))

# マーカの辞書作成
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

threshold = 30 # 未検出を許容する回数
idMax = 7 # 検出するべきマーカーの数

# フレームを表示する
def show(frame):
    # 0.5倍のサイズで表示する
    magnification = 0.5
    frame = cv2.resize(frame , (int(width * magnification), int(height * magnification)))
    cv2.imshow('frame', frame)
    cv2.waitKey(1) 

# エラー表示を追加する
def appendError(frame):
    errorImg = cv2.imread('errorImg.png')
    errorImg = cv2.resize(errorImg , (width, height))
    alpha = 0.3
    beta = 0.7
    return cv2.addWeighted(frame, alpha, errorImg, beta, 0)

# インデックスを指定してマーカーの中心座標を取得する
def getMarkerMean(ids, corners, index):
    for i, id in enumerate(ids):
        # マーカーのインデックス検索
        if(id[0] == index):
            v = np.mean(corners[i][0],axis=0) # マーカーの四隅の座標から中心の座標を取得する
            return [v[0],v[1]]
    return None

# 基準となるマーカーの取得
def getBasisMarker(ids, corners):
    # 左上、右上、左下、右下の順にマーカーの「中心座標」を取得
    basis = []
    basis.append(getMarkerMean(ids, corners, 1))
    basis.append(getMarkerMean(ids, corners, 2))
    basis.append(getMarkerMean(ids, corners, 3))
    basis.append(getMarkerMean(ids, corners, 4))
    return basis

# 監視エリアの取得
def getTargetArea(basis):
    # 取得座標の補正(Y座標で30%〜90%のエリアを取得)
    target = []
    x = basis[0][0] + (basis[2][0] - basis[0][0]) * 0.3
    y = basis[0][1] + (basis[2][1] - basis[0][1]) * 0.3
    target.append([x,y])
    x = basis[1][0] + (basis[3][0] - basis[1][0]) * 0.3
    y = basis[1][1] + (basis[3][1] - basis[1][1]) * 0.3
    target.append([x,y])
    x = basis[2][0] - (basis[2][0] - basis[0][0]) * 0.1
    y = basis[2][1] - (basis[2][1] - basis[0][1]) * 0.1
    target.append([x,y])
    x = basis[3][0] - (basis[3][0] - basis[1][0]) * 0.1
    y = basis[3][1] - (basis[3][1] - basis[1][1]) * 0.1
    target.append([x,y])
    return target

# 画像の変形
def getTransformImage(target, frame,  width, height):
    frame_coordinates = np.float32(target)
    target_coordinates   = np.float32([[0, 0],[width, 0],[0, height],[width, height]])
    trans_mat = cv2.getPerspectiveTransform(frame_coordinates,target_coordinates)
    return cv2.warpPerspective(frame, trans_mat, (width, height))

def main():

    errorCounter = 0
    markerError = False
    ColorCyan = (255, 255, 0)

    while True:
        # フレーム取得
        ret, frame = cap.read() 
        # マーカ検出
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary) 
        # マーカの必要数が検出されていない場合は、処理しない
        if ids is None:
            continue
        detectCount = len(ids)
        if(detectCount != idMax):
            print("detect:{} errorCounter:{}".format(detectCount, errorCounter))
            errorCounter += 1
            # 未検出が、規定数を超えたら、エラー画面にする
            if(errorCounter > threshold):
                markerError = True
            # エラー画面表示
            if(markerError == True):
                aruco.drawDetectedMarkers(frame, corners, ids, ColorCyan) 
                show(appendError(frame))
            continue

        # 以下、必要数が検出された場合の処理
        errorCounter=0
        markerError = False

        # 基準となる四隅のマーカーを取得
        basis =  getBasisMarker(ids, corners)

        # 監視エリアの取得
        target = getTargetArea(basis)

        # 監視対象の画像を変形して表示
        targetFrame = getTransformImage(target, frame, 1000, 600)
        cv2.imshow('targetFrame', targetFrame)

        # 基準線の描画
        cv2.line(frame, (basis[0][0], basis[0][1]), (basis[1][0], basis[1][1]), ColorCyan, thickness=1, lineType=cv2.LINE_4)
        cv2.line(frame, (basis[0][0], basis[0][1]), (basis[2][0], basis[2][1]), ColorCyan, thickness=1, lineType=cv2.LINE_4)
        cv2.line(frame, (basis[0][0], basis[0][1]), (basis[3][0], basis[3][1]), ColorCyan, thickness=1, lineType=cv2.LINE_4)
        #マーカ描画
        aruco.drawDetectedMarkers(frame, corners, ids, ColorCyan) 

        show(frame)

    cap.release() 
    cv2.destroyAllWindows() 

main()