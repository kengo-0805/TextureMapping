import os

import cv2
from PIL import Image
import pyglet
import pyglet.gl as gl
import pyrealsense2 as rs

import math
import numpy as np
import time

start = time.time()
# ===============================
# 定数
# ===============================
DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(os.path.dirname(__file__), DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.makedirs(DATA_DIRPATH)

CHESS_HNUM = 7  # 水平方向個数
CHESS_VNUM = 10  # 垂直方向個数
CHESS_MARGIN = 50  # [px]
CHESS_BLOCKSIZE = 80  # [px]

BOARD_WIDTH = 0.33  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
BOARD_X = 0.  # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.  # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = 1.5  # chessboard の3次元位置Z座標 [m]（右手系）

# OpenGL の射影のパラメータ
class Params:
    def __init__(self, zNear=0.0001, zFar=20.0, fovy=21.0):
        self.Z_NEAR = zNear  # 最も近い点 [m]
        self.Z_FAR = zFar  # 最も遠い点 [m]
        self.FOVY = fovy  # 縦の視野角 [deg]


PARAMS = Params(zNear=0.0001,  # [m]
                zFar=20.0,  # [m]
                fovy=21.0  # [deg]
                )

'''
# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
'''

# ===============================
# グローバル変数
# ===============================
window = None  # pyglet の Window　クラスのインスタンス
state = None  # アプリの状態を管理する変数（AppState）
cam_w, cam_h = 0, 0  # 画面解像度


class AppState:
    def __init__(self, params):
        self.params = params
        self.pitch = math.radians(0)
        self.yaw = math.radians(0)
        self.translation = np.array([0, 0, 0], np.float32)
        self.distance = 0
        self.lighting = False
        self.zNear = self.params.Z_NEAR




# テクスチャの自作
def make_chessboard(num_h, num_v, margin, block_size):
    chessboard = np.ones((block_size * num_v + margin * 2, block_size * num_h + margin * 2, 3), dtype=np.uint8) * 255

    for y in range(num_v):
        for x in range(num_h):
            if (x + y) % 2 == 0:
                sx = x * block_size + margin
                sy = y * block_size + margin
                chessboard[sy:sy + block_size, sx:sx + block_size, 0] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 1] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 2] = 0
    return chessboard




# 描画する画像の用意
def load_chessboard():
    global texture_ids, chessboard_image

    chessboard = make_chessboard(CHESS_HNUM, CHESS_VNUM, CHESS_MARGIN, CHESS_BLOCKSIZE)

    filepath = os.path.join(DATA_DIRPATH, 'chessboard.png')
    cv2.imwrite(filepath, chessboard)
    chessboard_image = Image.open(filepath)

    tw, th = chessboard_image.width, chessboard_image.height
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, tw, th, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, chessboard_image.tobytes())



#   座標の指定と描画
def board():
    global chessboard_image, texture_ids

    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)

    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(0.5 / chessboard_image.width, 0.5 / chessboard_image.height, 0)
    # テクスチャの頂点の指定
    gl.glBegin(gl.GL_QUADS)

# 座標たち
    gl.glTexCoord2i(0, 0) # テクスチャ座標（左上）id = 0
    gl.glVertex3f(-0.4766671 , -0.83889977 , 2.30024727) # オブジェクト座標
    gl.glTexCoord2i(0, 1) #　テクスチャ座標（左下）id = 3
    gl.glVertex3f(-0.49529457 ,-0.22799575 , 2.38920727)
    gl.glTexCoord2i(1, 1) # テクスチャ座標（右下）id = 2
    gl.glVertex3f(-0.06795887, -0.21453991 , 2.36340399)
    gl.glTexCoord2i(1, 0) # テクスチャ座標（右上）　8id = 1
    gl.glVertex3f(-0.05531667 ,-0.83239795 , 2.28709619)
    gl.glEnd()
    gl.glPopMatrix()


# 描画の世界を作っている
def on_draw_impl():
    window.clear()

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    width, height = window.get_size()
    # print(width,height)
    # width = 2560
    # height = 1600
    gl.glViewport(0, 0, width, height)

    # 射影行列の設定
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    aspect = width / float(height * 2)
    bottom = 0
    top = state.zNear * np.tan(np.radians(PARAMS.FOVY))
    left = - top * aspect
    right = top * aspect
    gl.glFrustum(left, right, bottom, top, state.zNear, PARAMS.Z_FAR)  # 視野錐台の大事なやつ

    # 視点の設定
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.gluLookAt(0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0,
                 0.0, -1.0, 0.0)

    gl.glTranslatef(0, 0, state.distance)
    gl.glRotated(state.pitch, 1, 0, 0)
    gl.glRotated(state.yaw, 0, 1, 0)

    gl.glTranslatef(0, 0, -state.distance)
    gl.glTranslatef(*state.translation)
    # * は分解して渡すことを意味している
    # gl.glTranslatef(*[a,b,c]) は gl.glTranslatef(a,b,c) と同じ

    if state.lighting:
        ldir = [0.5, 0.5, 0.5]  # world-space lighting
        ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
        ldir = list(ldir) + [0]  # w=0, directional light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     (gl.GLfloat * 3)(1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     (gl.GLfloat * 3)(0.75, 0.75, 0.75))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

    # comment this to get round points with MSAA on
    gl.glEnable(gl.GL_POINT_SPRITE)
    board()

'''
# -------------------------------
# ArUcoの読み取り箇所
# -------------------------------

aruco = cv2.aruco
# マーカーの辞書選択
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)
# マーカーのサイズ
marker_length = 0.056 # [m]

camera_matrix = np.load("mtx.npy")
distortion_coeff = np.load("dist.npy") #np.array(([0,0,0,0,0]))


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
            kado = corners[0]
            print("id ={}".format(ids[i]))
            print(tvec)
            print("corners={}".format(corners[0]))
            print("corners={}".format(corners[0][0]))
            print("corners={}".format(corners[0][0][0]))
            
            if ids[i] == 0:
                print("0を認識")
                f = open("id0.txt","w")
                f.write("{}".format(tvec))
                f.close()
              
            if ids[i] == 1:
                print("1を認識")
                n = open("id1.txt","w")
                n.write("{}".format(tvec))
                n.close()
                

capture.release()
cv2.destroyAllWindows()

'''

# -------------------------------
# ここからがメイン部分
# -------------------------------

texture_ids = (pyglet.gl.GLuint * 1) ()
gl.glGenTextures(1, texture_ids)
load_chessboard()


# アプリクラスのインスタンス
state = AppState(PARAMS)
platform = pyglet.window.get_platform()
display = platform.get_default_display()
screens = display.get_screens()
window = pyglet.window.Window(
    config=gl.Config(
        double_buffer=True,
        samples=8  # MSAA
    ),
    resizable=True,
    vsync=True,
    fullscreen=True,
    screen=screens[0])

@window.event
def on_draw():
    on_draw_impl()
pyglet.app.run()

elapsed_time = time.time() - start
# if @window.event = 1
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")