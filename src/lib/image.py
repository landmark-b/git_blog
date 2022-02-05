# %%
import os
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
# import albumentations as albu

def img_scaler(img:np.array) -> np.array:
    """ 0~255の範囲にスケーリングする
    Args:
        img (np.array): 入力画像
    Returns:
        np.array: スケーリング画像
    Note:
        画像である必要はないが、array全体でスケーリングされる点に注意。
    """
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    return img


def canny(img:np.ndarray, low_threshold:int, high_threshold:int)-> np.ndarray:
    """ Applies the Canny transform
    Args:
        img (np.ndarray): グレースケール画像
        low_threshold (int): minVal
        high_threshold (int): maxVal

    Returns:
        np.ndarray: エッジ画像
    Note: https://docs.opencv.org/4.5.5/da/d22/tutorial_py_canny.html
    """
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def grayscale(img, is_bgr=False):
    """Applies the Grayscale transform
    Args:
        img (np.ndarray): 画像
        is_bgr (bool, optional): カラー表現がBGRか. Defaults to False.
    Note:
        OpenCVで画像ファイルをReadした場合はBGR形式で読まれ、
        pltはRGB形式で処理するため、変換が必要。
    """

    if is_bgr:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def calc_region(imshape, left_bottom_rate, left_top_rate, right_top_rate, right_bottom_rate):
    """ マスク画像の４点を指定。画像は左上が原点

    Args:
        imshape (list): 元画像のshape。ただし[W, H]に変換している。
        left_bottom_rate ([list]): 画像に対する、左下の点の割合
        left_top_rate ([type]):  画像に対する、左上の点の割合
        right_top_rate ([type]):  画像に対する、右上の点の割合
        right_bottom_rate ([type]):  画像に対する、右下の点の割合

    Returns:
        [list of list]: マスク領域の４点を示すリスト[[w1,h1],[w2,h2],[w3,h3],[w4,h4]]
    """

    left_bottom = imshape * np.array(left_bottom_rate)
    left_top = imshape * np.array(left_top_rate)
    right_top = imshape * np.array(right_top_rate)
    right_bottom = imshape * np.array(right_bottom_rate)

    region_coord = [left_bottom, left_top, right_top, right_bottom] # 先行車領域の座標4点(左下から時計回り)

    return region_coord


# TODO 要改善
# 線分の延長を行う。.Append、や　np.poly1d、　np.polyfit(x,y,n)を利用した効率化が必要
# np.polyfit(x,y,n): n次式で２変数x,yの回帰分析
def draw_ext_lines(img, lines, color=[255, 0, 0], thickness=2):
    d = 300 # required extend length
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2 != x1):
                slope = (y2-y1)/(x2-x1)
                sita = np.arctan(slope)
                if (slope > 0): # 傾きに応じて場合分け
                    if (x2 > x1):
                        x3 = int(x2 + d*np.cos(sita))
                        y3 = int(y2 + d*np.sin(sita))
                        cv2.line(img, (x3, y3), (x1, y1), color, thickness)
                    else:
                        x3 = int(x1 + d*np.cos(sita))
                        y3 = int(y1 + d*np.sin(sita))
                        cv2.line(img, (x3, y3), (x2, y2), color, thickness)
                elif (slope < 0):
                    if (x2 > x1):
                        x3 = int(x1 - d*np.cos(sita))
                        y3 = int(y1 - d*np.sin(sita))
                        cv2.line(img, (x3, y3), (x2, y2), color, thickness)
                    else:
                        x3 = int(x2 - d*np.cos(sita))
                        y3 = int(y2 - d*np.sin(sita))
                        cv2.line(img, (x3, y3), (x1, y1), color, thickness)

def hough_lines(img, rho=2, threshold=2, min_line_len=200, max_line_gap=10, theta=np.pi/180):
    """
    `img` should be the output of a Canny transform.
    Args:
        rho: distance resolution in pixels of the Hough grid
        theta: angular resolution in radians of the Hough grid
        threshold: minimum number of votes (intersections in Hough grid cell)
        min_line_len: minimum number of pixels making up a line
        max_line_gap: maximum gap in pixels between connectable line segments

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ###draw_lines(line_img, lines)
    if (not isinstance(lines,type(None))):
        draw_ext_lines(line_img, lines)

    return line_img

def pipeline_calc_white_lines(img:np.ndarray, param:dict, DEBUG=False) -> np.ndarray:
    """ 道路の車線(white line)を検出する

    Args:
        img (np.ndarray): 道路の走行画像
        param (dict): パラメータファイル

    Returns:
        [np.ndarray]: imgと同じshapeの車線検出画像
    """

    # gray & edge detect
    gray_img = grayscale(img)
    edges = canny(gray_img, **param['canny'])
    if DEBUG:
        plt_imshow(edges)

    # create a masked edges image
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    ## define a four sided polygon to mask
    mask = np.zeros_like(edges)
    # TODO 白線検出領域の設定値の決め方
    region_coord = calc_region(np.array(gray_img.shape)[::-1], **param['region_rate'])
    vertices = np.array([region_coord], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # mask edges by region
    masked_edges = cv2.bitwise_and(edges, mask)
    if DEBUG:
        plt_imshow(masked_edges)

    # hough line detection
    # TODO thetaをパラメータyamlでどうやって扱うか
    # TODO パラメータを画像毎に最適化する必要があるかも。今は決め打ち。
    line_img = hough_lines(masked_edges, theta=np.pi/180, **param['hough']) # Lineのみの画像
    if DEBUG:
        plt_imshow(line_img)

    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """ overlay img
    output = initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """

    return cv2.addWeighted(initial_img, a, img, b, c)


def draw_bbox(img:np.ndarray, bbox:np.ndarray, clr=100, thickness=1):
    """ bboxをimgに描画
    Args:
        img (np.ndarray): image
        bbox (np.ndarray): [x, y, w, h]形式
        clr (int or list): color
    """
    x, y, w, h  = [int(v) for v in bbox]
    img = cv2.rectangle(img, (x,y), (int(x+w), int(y+h)), clr, thickness)

    return img

def plt_imshow(img:np.ndarray, is_bgr=False, cmap='gray', title=None):
    """ 画像を表示

    Args:
        img (np.ndarray): 画像のndaray
        is_bgr (bool, optional): カラー表現がBGRか. Defaults to False.
    Note:
        OpenCVで画像ファイルをReadした場合はBGR形式で読まれ、
        pltはRGB形式で処理するため、変換が必要。
    """

    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, ax = plt.subplots()
    if title:
        ax.set_title(title, fontsize=16, color='white')
    ax.imshow(img, cmap=cmap)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

class Video():
    def __init__(self, video_path:Path):
        self.cap = cv2.VideoCapture(str(video_path))
        assert self.cap.isOpened(), f"video path {video_path} cannot be opened"

        self.update_video_configs()

    def update_video_configs(self):
        self.confs = {}

        self.confs['frame_width'] = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) # ビデオストリームの幅
        self.confs['frame_height'] = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # ビデオストリームの高さ
        self.confs['fps'] = self.cap.get(cv2.CAP_PROP_FPS) # FPS
        self.confs['frame_count'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数
        self.confs['pos_frames'] = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) # 現在のフレーム
        self.confs['pos_msec'] = self.cap.get(cv2.CAP_PROP_POS_MSEC) # 経過時間

    def set_frame_num(self, frame_num:int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


    def read_frame_as_array(self, frame_num:int):
        """ read specified video frame as ndarray

        Args:
            frame_num (int, optional): frame number to read.
        """

        self.set_frame_num(frame_num)

        ret, frame = self.cap.read()
        assert ret, f"can not read frame. frame number: {frame_num}"

        return frame


    def dump_each_frames(self, o_path:Path, frame_num=None, is_parent=True):
        """ dump each frame of video, as image file.

        Args:
            o_path ([Path]): output image file path&name
            frame_num (int): frame number to output
        Note:
            current frame number(cv2.CAP_PROP_POS_FRAMES) is set to 'frame_num'
        Exsample:
            - dump all frames
            for num_frame in range(int(cap.confs['frame_count'])):
                cap.dump_each_frames(output_dir / f'{num_frame:08d}f.png', num_frame)
        """

        frame = self.read_frame_as_array(frame_num)
        o_path.parent.mkdir(exist_ok=True, parents=is_parent)
        cv2.imwrite(str(o_path), frame)

    def dump_all_frames(self, o_dir:Path, is_parent=True):
        """ dump all frame of video, as image file.

        Args:
            o_dir (Path): png file output directory
            is_parent (bool, optional): [description]. Defaults to True.
        """

        o_dir.mkdir(exist_ok=True, parents=is_parent)

        for frame_num in range(self.confs['frame_count']):
            frame = self.read_frame_as_array(frame_num)
            cv2.imwrite(str(o_dir / f'{int(frame_num):08d}f.png'), frame)
