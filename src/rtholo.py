from utils import *
from propagation_ASM import *
from NET1 import NET1
from CNN import *
from CNN_PP import *
from rich.progress import track



class rtholo(nn.Module):

    def __init__(self,size, mode = 'train', feature_size = 7.48e-6, distance_range = 0.03, 
                 img_distance = 0.2,layers_num = 30,num_layers = 10, num_filters_per_layer=15, CNNPP = False):
        super().__init__()

        self.network1 = NET1()
        if CNNPP == False:
            self.network2 = CNN(input_dim=2, output_dim=1, num_layers=num_layers, num_filters_per_layer=num_filters_per_layer, filter_size=3)
        else:
            self.network2 = CNN_PP(input_dim=8, output_dim=4, num_layers=num_layers, num_filters_per_layer=num_filters_per_layer, 
                                    filter_size=3, img_size=size)

        self.wavelength = 632e-9
        self.feature_size = [feature_size, feature_size]
        self.z = img_distance
        self.precomputed_H = None
        self.return_H = None
        self.size = size
        self.distance_range = distance_range
        self.layers_num = layers_num
        self.img_distance = img_distance

        print('====================================================')
        print('network1:{}'.format(self.network1.__class__.__name__))
        print('network2:{}'.format(self.network2.__class__.__name__))
        print('feature_size:{}'.format(self.feature_size))
        print('z:{}'.format(self.z))
        print('size:{}'.format(self.size))
        print('distance_range:{}'.format(self.distance_range))
        print('layers_num:{}'.format(self.layers_num))
        print('img_distance:{}'.format(self.img_distance))
        print('====================================================')

        if self.precomputed_H == None:
            self.precomputed_H = propagation_ASM(torch.empty(1, 1, self.size, self.size), feature_size=[feature_size, feature_size],
                                                 wavelength=self.wavelength, z=self.z, return_H=True)
            self.precomputed_H = self.precomputed_H.to('cuda').detach()
            self.precomputed_H.requires_grad = False

        self.mode = mode

        if self.mode == 'train':
            self.pre_kernel = []
            for i in track(range(layers_num)):

                distance = (0-self.distance_range)/self.layers_num*i
                dis = distance - self.img_distance
                if isinstance(dis,torch.Tensor):
                    dis = dis.item()
                dis = round(dis, 6)


                self.pre_kernel.append(propagation_ASM(torch.empty(1, 1, self.size, self.size), feature_size=[feature_size, feature_size],
                                            wavelength=self.wavelength, z=dis, return_H=True))
                self.pre_kernel[i] = self.pre_kernel[i].to('cuda').detach()
                # self.pre_kernel[i] = self.pre_kernel[i].to('cpu').detach()
                self.pre_kernel[i].requires_grad = False



    def forward(self, source, ikk):

        target_amp, target_phase = self.network1(source)        
        obj_r, obj_i = polar_to_rect(target_amp, target_phase)
        target_field = torch.complex(obj_r, obj_i)

        slm_field = propagation_ASM(target_field, self.feature_size, self.wavelength, self.z, precomped_H=self.precomputed_H)
       
        slm_amp, slm_phase = rect_to_polar(slm_field.real, slm_field.imag)
        slm_field = torch.cat([slm_amp, slm_phase], dim=-3)        
        
        
        holo = self.network2(slm_field)
        
        if self.mode == 'train':
            H_real, H_imag = polar_to_rect(torch.ones(holo.shape).cuda(), holo)
            #                              对振幅进行限制，只保留相位信息
            holo_field = torch.complex(H_real, H_imag)

            distance = (0-self.distance_range)/self.layers_num*ikk
            dis = distance - self.img_distance
            if isinstance(dis,torch.Tensor):
                dis = dis.item()
            dis = round(dis, 6)

            if isinstance(ikk,torch.Tensor):
                ikk = ikk.item()
        
            recon_field = propagation_ASM(holo_field, self.feature_size, self.wavelength, dis, precomped_H=self.pre_kernel[ikk])
            # recon_field = propagation_ASM(holo_field, self.feature_size, self.wavelength, dis, precomped_H=self.pre_kernel[ikk].to('cuda'))
        else:
            recon_field = 0

        return holo, slm_amp, recon_field
    
    def get_Network1(self):
        return self.network1
    
    def get_Network2(self):
        return self.network2

