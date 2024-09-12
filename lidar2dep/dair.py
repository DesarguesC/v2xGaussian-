import pdb

BASE_DIR = './cooperative-vehicle-infrastructure'
# TODO: read DAIR-V2X dataset
import os, json, cv2, re, pdb
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from lidar2dep.main import get_CompletionFormer as getFormer
from lidar2dep.main import create_former_input
from PIL import Image
from cam_utils import ab64, downsampler



def load_camera_intrinsic(json_path, downsample: int = 1, return_dict: bool = False):
    # ./infrastructure-side/calib/camera_intrinc
    """

        RETURN:
            width, height, fx, fy, cx, cy
        cam_K:
             _         _
            |  fx 0  cx |
            |  0  fy cy |
            |_ 0  0  1 _|

        cam_D:
            畸变参数k1, k2, k3, p1, p2

    """
    with open(json_path) as f:
        ex = json.load(f)
    # camera_id = ex['cameraID']
    print(f'ex is None: {ex is None}')
    h, w = ex['height'], ex['width']
    h, w = ab64((h // downsample, w // downsample))
    # projection = ex['P']
    # cam_D, cam_K = ex['cam_D'], ex['cam_K']
    # cam_D = np.array(ex['cam_D'], dtype=np.float32)
    cam_K = np.array(ex['cam_K'], dtype=np.float32)
    cam_K = np.array([cam_K[0:3], cam_K[3:6], cam_K[6:9]])
    
    if return_dict:
        return {
            'intrinsic': {
                "width": int(w), "height": int(h), 
                "fx": float(cam_K[0,0]), "fy": float(cam_K[1,1]), 
                "cx": float(cam_K[0,2]), "cy": float(cam_K[1,2])
            },
            'extrinsic': np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.]]),  # ?
            'intrinsic_matrix': cam_K
        }
    else:
        return int(w), int(h), float(cam_K[0,0]), float(cam_K[1,1]), float(cam_K[0,2]), float(cam_K[1,2])

def load_camera_R(json_path):
    # ./infrastructure-side/calib/virtuallidar_to_camera
    with open(json_path) as f:
        ex = json.load(f)
    R = np.array(ex['rotation'])
    T = np.array(ex['translation'])
    A = np.zeros((4,4))
    A[0:3,0:3] = R
    A[0:3,-1] = np.squeeze(T)
    A[-1,-1] = 1.
    return A # np.array[4, 4]: [R, t]

def render_pcd_view_img(pcd, c_i, c_e, shape):
    """
        pcd: pcd data
        c_i: intrinsic
        c_e: extrinsic dict:
            {
                'height': ..., 'width': ...,
                'fx': ..., 'cx': ..., 'fy': ..., 'cy': ...
            }
        shape: image shape [H W 3]
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=shape[1], height=shape[0])
    vis.add_geometry(pcd)
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = c_e
    camera_parameters.intrinsic.set_intrinsics(**c_i)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_parameters)
    # -> [Open3D WARNING] [ViewControl] ConvertFromPinholeCameraParameters() failed because window height and width do not match.
    vis.poll_events()
    vis.update_renderer()

    return np.asarray(vis.capture_screen_float_buffer(do_render=True))



def lidar2word(json_path):
    pass


class DAIR_V2X_C:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.config_path = os.path.join(base_dir, 'cooperative-vehicle-infrastructure/cooperative/data_info.json')
        with open(self.config_path) as f:
            self.items = json.load(f)
    def __getitem__(self, idx, require_path=False):
        return self.items[idx] if not require_path else (self.items[idx], self.base_dir)
    def __len__(self):
        return len(self.items)
    
class CooperativeData:
    def __init__(self, dair, path: str = None, downsample: int = 1):
        # When initailizing, set path as the base direction where DAIR-V2X locates
        # dair: DAIR_V2X_C[idx]
        self.downsample = int(downsample)

        if path is None:
            path = '..'
        self.model_path = path
        self.inf_id = re.split('[/\.]', dair['infrastructure_image_path'])[-2]
        self.veh_id = re.split('[/\.]', dair['vehicle_image_path'])[-2]
        #
        # self.inf_side_img = cv2.imread(os.path.join('./cooperative-vehicle-infrastructure-infrastructure-side-image', f'{inf_id}.jpg'))
        # # print(os.path.join(BASE_DIR, dair['infrastructure_image_path']))
        # self.height, self.width, _ = self.inf_side_img.shape
        # self.inf_side_pcd = o3d.io.read_point_cloud(os.path.join('./cooperative-vehicle-infrastructure-infrastructure-side-velodyne', f'{inf_id}.jpg'))
        #
        # self.veh_side_img = cv2.imread(os.path.join('cooperative-vehicle-infrastructure-vehicle-side-image', f'{veh_id}.jpg'))
        # self.veh_side_pcd = o3d.io.read_point_cloud(os.path.join('./cooperative-vehicle-infrastructure-vehicle-side-velodyne', f'{veh_id}.jpg'))
        #
        # with open(os.path.join(BASE_DIR, dair['cooperative_label_path'])) as f:
        #     self.labels_list = json.load(f) # list of dict
        #
        # self.offset_dict = dair['system_error_offset']
        #
        u = load_camera_intrinsic(os.path.join(f'{path}/cooperative-vehicle-infrastructure/infrastructure-side/calib/camera_intrinsic', f'{self.inf_id}.json'), downsample=self.downsample, return_dict=True)
        self.camera_intrinsic = u['intrinsic']
        self.inf_ex = u['extrinsic']
        self.camera_intrinsic_matrix = u['intrinsic_matrix']
        self.veh_ex = load_camera_R(os.path.join(f'{path}/cooperative-vehicle-infrastructure/vehicle-side/calib/lidar_to_camera', f'{self.veh_id}.json'))


        # inf/veh rgb image file path
        self.inf_img_path = f'{path}/cooperative-vehicle-infrastructure-infrastructure-side-image/{self.inf_id}.jpg'
        self.veh_img_path = f'{path}/cooperative-vehicle-infrastructure-vehicle-side-image/{self.veh_id}.jpg'
        self.inf_pcd_path = f'{path}/cooperative-vehicle-infrastructure-infrastructure-side-velodyne/{self.inf_id}.pcd'
        self.veh_pcd_path = f'{path}/cooperative-vehicle-infrastructure-vehicle-side-velodyne/{self.veh_id}.pcd'
        # camera intrinsics matrix path
        self.inf_cam_intrinsic_path = f'{path}/cooperative-vehicle-infrastructure/infrastructure-side/calib/camera_intrinsic/{self.inf_id}.json'
        self.veh_cam_intrinsic_path = f'{path}/cooperative-vehicle-infrastructure/vehicle-side/calib/camera_intrinsic/{self.veh_id}.json'
        # inf/veh lidar2camera transformation matrix path
        self.inf_lidar2cam_path = f'{path}/cooperative-vehicle-infrastructure/infrastructure-side/calib/virtuallidar_to_camera/{self.inf_id}.json'
        self.veh_lidar2cam_path = f'{path}/cooperative-vehicle-infrastructure/vehicle-side/calib/lidar_to_camera/{self.veh_id}.json'
        # inf lidar2world matrix path
        self.inf_lidar2world_path = f'{path}/cooperative-vehicle-infrastructure/infrastructure-side/calib/virtuallidar_to_world/{self.inf_id}.json'
        # veh lidar2world matrix path
        self.veh_lidar2novatel_path = f'{path}/cooperative-vehicle-infrastructure/vehicle-side/calib/lidar_to_novatel/{self.veh_id}.json'
        self.veh_novatel2world_path = f'{path}/cooperative-vehicle-infrastructure/vehicle-side/calib/novatel_to_world/{self.veh_id}.json'

        # DAIR: ../DAIR-V2X/...
        # PLY: ../ply

        inf_path_list = ['inf', str(self.inf_id), 'ply']
        inf_init = f'{path}/../'
        for pp in inf_path_list:
            inf_init = os.path.join(inf_init, pp)
            if not os.path.exists(inf_init): os.mkdir(inf_init)
        self.inf_ply_store_path = os.path.join(inf_init, f'{self.inf_id}.ply')

        veh_path_list = ['veh', str(self.veh_id), 'ply']
        veh_init = f'{path}/../'
        for pp in veh_path_list:
            veh_init = os.path.join(veh_init, pp)
            if not os.path.exists(veh_init): os.mkdir(veh_init)
        self.veh_ply_store_path = os.path.join(veh_init, f'{self.veh_id}.ply')

        try:
            self.inf_side_img = Image.open(self.inf_img_path)
            self.veh_side_img = Image.open(self.veh_img_path)
        except Exception as err:
            print(f'err: {err}')
            pdb.set_trace()

        self.inf_side_img = downsampler(self.inf_side_img, self.downsample)
        self.veh_side_img = downsampler(self.veh_side_img, self.downsample)

    def set_downsample(self, downsample):
        self.downsample = downsample
        u = load_camera_intrinsic(
            os.path.join(f'{self.model_path}/cooperative-vehicle-infrastructure/infrastructure-side/calib/camera_intrinsic', f'{self.inf_id}.json'), downsample=self.downsample, return_dict=True)
        self.camera_intrinsic = u['intrinsic']
        # self.camera_intrinsic['width'] = ab64(self.camera_intrinsic['width'])
        # self.camera_intrinsic['height'] = ab64(self.camera_intrinsic['height'])
        self.inf_side_img = downsampler(self.inf_side_img, self.downsample)
        self.veh_side_img = downsampler(self.veh_side_img, self.downsample)



    # PREVIOUS API - <Begin>
    def show_side(self, which='inf'):
        assert which in ['inf', 'veh'], which
        fig = plt.figure()
        # infrastructure-view
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(getattr(self, f'{which}_side_img'), cv2.COLOR_RGB2BGR))
        ax1.set_title(f'{which}-side')

        ax2 = fig.add_subplot(1, 2, 2)

        side_pcd_img = render_pcd_view_img(
                getattr(self, f'{which}_side_pcd'), 
                self.camera_intrinsic, 
                getattr(self, f'{which}_ex'), 
                getattr(self, f'{which}_side_img').shape
            )
        ax2.imshow(cv2.cvtColor(side_pcd_img, cv2.COLOR_RGB2BGR))
        ax2.set_title('rendered-pcd')

        plt.tight_layout()
        plt.show()
        
        return getattr(self, f'{which}_side_img'), side_pcd_img
    def get_side(self, which='inf'):
        assert which in ['inf', 'veh'], which
        side_img = getattr(self, f'{which}_side_img')
        pcd_rendered = render_pcd_view_img(
                getattr(self, f'{which}_side_pcd'),
                self.camera_intrinsic,
                getattr(self, f'{which}_ex'),
                getattr(self, f'{which}_side_img').shape
            )
        return side_img, pcd_rendered
    def get_side_and_depth(self, opt, which='inf', former=None, w=0.3):
        assert which in ['inf', 'veh'], which
        if former is None:
            former = getFormer(opt)
        side_img = getattr(self, f'{which}_side_img')
        pcd_rendered = render_pcd_view_img(
            getattr(self, f'{which}_side_pcd'),
            self.camera_intrinsic,
            getattr(self, f'{which}_ex'),
            getattr(self, f'{which}_side_img').shape
        )
        depth = former(create_former_input(side_img, pcd_rendered, self.camera_intrinsic_matrix))
        try:
            weighted_depth = pcd_rendered * w + depth * (1. - w)
        except Exception as err:
            print(f'error: {err}')
            pdb.set_trace()

        return side_img, pcd_rendered, depth, weighted_depth

    # PREVIOUS API - <End>

    def load_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        arr = np.asarray(pcd.points)
        # print(arr.shape)
        new_arr = np.ones((arr.shape[0], 4))
        new_arr[:, 0:3] = arr

        assert new_arr.shape[-1] == 4, new_arr.shape
        # get a point: arr[:,idx]
        return pcd, new_arr.T

    def load_intrinsic(self, path, matrix_read_only = False, downsample: int = None):
        with open(path) as f:
            ff = json.load(f)
        cam_K = ff['cam_K']
        K = np.array([cam_K[0:3], cam_K[3:6], cam_K[6:9]])
        assert K.shape == (3, 3)
        if matrix_read_only: return K

        height = ff['height']
        width = ff['width']
        if downsample is None: downsample = self.downsample
        height, width = ab64((height // downsample, width // downsample))

        return height, width, K

    def intrinsic2dict(self, h, w, K):
        # print(f'h = {h}, w = {w}, K ={K}')
        assert isinstance(K, np.ndarray), f'type(K): {type(K)}'
        return {
            'height': h,
            'width': w,
            'fx': K[0, 0], 'cx': K[0, -1], 'fy': K[1, 1], 'cy': K[1, -1]
        }

    def load_extrinsic(self, path):
        # -> [R | t]
        with open(path) as f:
            ff = json.load(f)
        if 'transform' in ff:
            ff = ff['transform']
        R, T = np.array(ff['rotation']), np.array(ff['translation'])
        assert R.shape == (3, 3) and T.shape == (3, 1), f'R.shape = {R.shape}\nT.shape = {T.shape}'
        trans = np.zeros((4, 4))
        trans[0:3, 0:3] = R
        trans[0:3, -1] = np.squeeze(T)
        trans[-1, -1] = 1.
        return trans # [4 4] extrinsic matrix

    def load4pcd_render(self, type='inf'):
        assert type in ['inf', 'veh'], type
        if type == 'inf':
            h, w, K = self.load_intrinsic(self.inf_cam_intrinsic_path)
        else:
            # TODO: 确认一下是不是车端路端的相机内参一致
            h, w, _ = self.load_intrinsic(self.inf_cam_intrinsic_path)
            K = self.load_intrinsic(self.veh_cam_intrinsic_path, matrix_read_only=True)
        intrinsic_dict = self.intrinsic2dict(h, w, K)
        extrinsic_array = self.load_extrinsic(getattr(self, f'{type}_lidar2cam_path'))

        return {
                'intrinsic': {
                    'dict': intrinsic_dict, # intrinsics dict
                    'matrix': K # [3 3] array
                    },
                'extrinsic': extrinsic_array, # [4 4] array
        }

    # 全都放到world坐标系的方法
    def inf_side_pcd2world(self):
        _, inf_pcd_arr = self.load_pcd(self.inf_pcd_path)
        Trans = self.load_extrinsic(self.inf_lidar2world_path)  # [4 4]
        pcd_world = Trans @ inf_pcd_arr
        return pcd_world[0:3, :]

    def veh_side_pcd2world(self):
        _, veh_pcd_arr = self.load_pcd(self.veh_pcd_path)
        Trans1 = self.load_extrinsic(self.veh_lidar2novatel_path)
        Trans2 = self.load_extrinsic(self.veh_novatel2world_path)
        pcd_world = Trans2 @ Trans1 @ veh_pcd_arr
        return pcd_world[0:3, :]

    def world2inf_cam(self):
        return np.linalg.inv(self.load_extrinsic(self.inf_lidar2world_path)) \
                        @ self.load_extrinsic(self.inf_lidar2cam_path)

    def world2veh_cam(self):
        return np.linalg.inv(self.load_extrinsic(self.veh_lidar2novatel_path) @ \
                             self.load_extrinsic(self.veh_novatel2world_path)) \
                                    @ self.load_extrinsic(self.veh_lidar2cam_path)




