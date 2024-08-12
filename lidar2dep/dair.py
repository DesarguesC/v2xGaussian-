BASE_DIR = './cooperative-vehicle-infrastructure'
# TODO: read DAIR-V2X dataset
import os, json, cv2, re
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from lidar2dep.main import get_CompletionFormer as getFormer
from lidar2dep.main import create_former_input


def load_camera_intrinsic(json_path, return_dict=False):
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
        self.config_path = os.path.join(base_dir, 'cooperative/data_info.json')
        with open(self.config_path) as f:
            self.items = json.load(f)
    def __getitem__(self, idx):
        return self.items[idx]
    def __len__(self):
        return len(self.items)
    
class CooperativeData:
    def __init__(self, dair):
        # dair: DAIR_V2X_C[idx]
        assert isinstance(dair, dict)
        inf_id = re.split('[/\.]', dair['infrastructure_image_path'])[-2]
        veh_id = re.split('[/\.]', dair['vehicle_image_path'])[-2]
        self.inf_side_img = cv2.imread(os.path.join('./cooperative-vehicle-infrastructure-infrastructure-side-image', f'{inf_id}.jpg'))
        # print(os.path.join(BASE_DIR, dair['infrastructure_image_path']))
        self.height, self.width, _ = self.inf_side_img.shape
        self.inf_side_pcd = o3d.io.read_point_cloud(os.path.join('./cooperative-vehicle-infrastructure-infrastructure-side-velodyne', f'{inf_id}.jpg'))
        
        self.veh_side_img = cv2.imread(os.path.join('cooperative-vehicle-infrastructure-vehicle-side-image', f'{veh_id}.jpg'))
        self.veh_side_pcd = o3d.io.read_point_cloud(os.path.join('./cooperative-vehicle-infrastructure-vehicle-side-velodyne', f'{veh_id}.jpg'))
        
        with open(os.path.join(BASE_DIR, dair['cooperative_label_path'])) as f:
            self.labels_list = json.load(f) # list of dict
        
        self.offset_dict = dair['system_error_offset']

        u = load_camera_intrinsic(os.path.join(f'{BASE_DIR}/infrastructure-side/calib/camera_intrinsic', f'{inf_id}.json'), return_dict=True)
        self.camera_intrinsic = u['intrinsic']
        self.inf_ex = u['extrinsic']
        self.camera_intrinsic_matrix = u['intrinsic_matrix']
        self.veh_ex = load_camera_R(os.path.join(f'{BASE_DIR}/vehicle-side/calib/lidar_to_camera', f'{veh_id}.json'))

        # self.camera_extrinsic_inf = 

        # self.camera_intrinsic_veh = load_camera_intrinsic(os.path.join(f'{BASE_DIR}/vehicle-side/calib/camera_intrinsic', f'{veh_id}.json'), return_dict=True)
        # self.camera_extrinsic_veh = load_camera_extrinsic(os.path.join(f'{BASE_DIR}/vehicle-side/calib/virtuallidar_to_camera', f'{veh_id}.json'))

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

    def get_side_and_depth(self, opt, which='inf', former=None):
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

        return side_img, pcd_rendered, depth
