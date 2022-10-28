import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # Set luminance to one
    points[:, 3] = 1

    return points


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            # Dictionary와 유사한 형식으로 추정
            key, value = line.split(':', 1)
            # 선행, 후행 문자만 지우는거
            value = value.strip()
            data[key] = value
            # value 의 문자가 다 float_chars에 들어가있으면
            if float_chars.issuperset(value):
                # try to cast to float array- 아래랑 연관지어 고민
                try:
                    # 위에는 str, 아래는 ndarray인데?
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    # File components as dictionary
    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub*(n-1)+colSub-1

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    # Camera intrinsic: 4x4
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    
    # get image shape
    im_shape=cam2cam['S_rect_02'][::-1]
    
    # compute projection matrix velodyne->image plane
    R_cam2rect=np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect=cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    # Cam image to rectangular frame
    # rectangular frame to velodyne image
    P_velo2im=np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance    
    velo=load_velodyne_points(velo_filename)
    velo=velo[velo[:, 0]>=0, :]
    
    # project the points to the camera
    velo_pts_im=np.dot(P_velo2im, velo.T).T
    velo_pts_im[:,:2]=velo_pts_im[:, :2]/ velo_pts_im[:, 2][..., np.newaxis]
    
    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

        