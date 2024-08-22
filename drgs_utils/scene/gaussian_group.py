from gaussian_model import *


class MultiGaussianModel(GaussianModel):

    def __init__(self, sh_degree):
        self.inf_gs = GaussianModel(sh_degree)
        self.veh_gs = GaussianModel(sh_degree)
        # self.compensate = GaussianModel(sh_degree) # 反演
        # TODO: 尝试一个以 inf&veh 点云反演变换产生的点云，作为初始化补偿GS
        # TODO: 先生成最小包围球Welzl算法(https://juejin.cn/post/7055284752361717791)，计算veh关于inf的的反演球






