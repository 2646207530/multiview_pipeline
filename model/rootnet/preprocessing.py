import numpy as np
import cv2
import random
from PIL import Image
from plyfile import PlyData, PlyElement
import torch

def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(np.uint8(cv_img), cv2.COLOR_BGR2RGB))

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    """
        Generate a transformed patch from a given image and bounding box.

        Args:
            cvimg (np.ndarray): The input image in OpenCV format (BGR).
            bbox (list or tuple): A bounding box represented as [x, y, width, height],
                where (x, y) are the coordinates of the top-left corner of the bounding box.
            scale (float): A scale factor for the bounding box transformation.
            rot (float): Rotation angle in degrees.
            do_flip (bool): Whether to flip the image horizontally.
            out_shape (tuple): The desired output shape of the patch as (height, width).

        Returns:
            tuple: A tuple containing:
                - img_patch (np.ndarray): The transformed patch image in OpenCV format (BGR) and float32 data type.
                - trans (np.ndarray): The affine transformation matrix used to generate the patch.
                - inv_trans (np.ndarray): The inverse affine transformation matrix.

        """
    img = cvimg.copy()  # FIXME: why copy?
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale,
                                    rot)  # (2, 3)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])),
                               flags=cv2.INTER_LINEAR)  # 应用上述仿射变换, (output_shape[1], output_shape[0], 3)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    """
        Rotate a 2D point by a given angle in radians.

        Args:
            pt_2d (list or np.ndarray): A 2D point represented as [x, y].
            rot_rad (float): The rotation angle in radians.

        Returns:
            np.ndarray: The rotated 2D point represented as [x', y'] in np.float32 data type.
        """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
        Generate an affine transformation matrix for transforming a patch from source dimensions to destination dimensions.

        Args:
            c_x (float): The x-coordinate of the center of the source patch.
            c_y (float): The y-coordinate of the center of the source patch.
            src_width (float): The width of the source patch.
            src_height (float): The height of the source patch.
            dst_width (float): The desired width of the destination patch.
            dst_height (float): The desired height of the destination patch.
            scale (float): The scale factor for scaling the source patch.
            rot (float): The rotation angle in degrees for rotating the source patch.
            inv (bool, optional): Whether to generate the inverse transformation matrix. Defaults to False.

        Returns:
            np.ndarray: A 2x3 float32 numpy array representing the affine transformation matrix.
        """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox


def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = bbox.astype(np.float32)
    return bbox

def convert_bbox(depth, rgb_image, depth_image, rgb_bbox):
    rgb_height, rgb_width, _ = rgb_image.shape
    depth_height, depth_width = depth_image.shape

    x1_rgb, y1_rgb, x2_rgb, y2_rgb = rgb_bbox

    center_x_rgb = (x1_rgb + x2_rgb) / 2
    center_y_rgb = (y1_rgb + y2_rgb) / 2

    center_x_depth = int(center_x_rgb * depth_width / rgb_width)
    center_y_depth = int(center_y_rgb * depth_height / rgb_height)

    depth_bbox_width = int((x2_rgb - x1_rgb) * depth_width / rgb_width)
    depth_bbox_height = int((y2_rgb - y1_rgb) * depth_height / rgb_height)

    # 修正深度边界框的坐标，确保不超出深度图像范围
    depth_x1 = max(0, center_x_depth - depth_bbox_width // 2)
    depth_y1 = max(0, center_y_depth - depth_bbox_height // 2)
    depth_x2 = min(depth_width, center_x_depth + depth_bbox_width // 2)
    depth_y2 = min(depth_height, center_y_depth + depth_bbox_height // 2)

    # 返回深度边界框的坐标
    return (depth_x1, depth_y1, depth_x2, depth_y2)