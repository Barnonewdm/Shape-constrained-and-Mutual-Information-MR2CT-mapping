import numpy as np
import md
import os
from PIL import Image
import matplotlib
import scipy.misc

# Data_root_3d = '/data0/geyunhao/MR2CT/ZS10307488/'
# Data_root_3d = '/data0/geyunhao/MR2CT_SAMPLING/ZS18111863/'
Data_root_3d = '/data0/geyunhao/MR2CT_registration/'
# Data_sam_root = '/data0/geyunhao/MR2CT_SAMPLING/'
# Data_root_3d = '/data0/geyunhao/MR2CT/'
Data_root_2d = '/home/geyunhao/Mapping/Mapping/CycleGAN_Lcc/datasets/MR2CT_reg/'
DATA_NAME = 'train' # train, test, val, ex
IMAGE_TYPE = '.png'
'''
    Turn the 3D slice to 2D slice and satisfy the structure of cycleGAN
'''
# save to .png

target_path_mr = Data_root_2d + DATA_NAME + 'A'
target_path_nct = Data_root_2d + DATA_NAME + 'B'
if not os.path.exists(target_path_mr):
    os.makedirs(target_path_mr)
if not os.path.exists(target_path_nct):
    os.makedirs(target_path_nct)


for roots, dirs, files in os.walk(Data_root_3d):
    for file in files:
        file_path = os.path.join(roots, file)
        if 'reT' in file: # MR
            try:
                mr_3d = md.read_image(file_path)  # (x,y,z)
                mr_3d_np = mr_3d.to_numpy() # (z,y,x)
                for i in range(mr_3d_np.shape[0]): # z
                    '''
                    get part of data
                    '''
                    # the approximately leg part in mri is range (0-60)
                    # the approximately pelvicum part in mri is range (70-230)
                    # the approximately lib part in mri is range (200-360)
                    # if i >= 70 and i <= 230:
                    '''
                    whole body
                    '''
                    # erase the first several slice (20) for the data blance
                    if i >= 20 :
                        slice0 = np.expand_dims(mr_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
                        # row data have been normalized and resized

                        target_path = target_path_mr
                        target_filename = 'mr_' + roots.split('/')[-1] + '_'+ str(i) + IMAGE_TYPE
                        if not os.path.exists(target_path + '/' + target_filename):
                            scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
            except:
                print(file_path)





        elif 'MR2CT_reg' in file:
            try:
                nct_3d = md.read_image(file_path)  # (x,y,z)
                nct_3d_np = nct_3d.to_numpy()  # (z,y,x)

                j = 0 #
                for i in range(nct_3d_np.shape[0]):  # z
                    '''
                    get part of data
                    '''
                    # the approximately leg part in nfct is range (0-60)
                    # the approximately pelvicum part in nfct is range (40-200)
                    # the approximately lib part in nfct is range (200-360)
                    # if i >= 40 and i <= 200:
                    '''
                    whole body
                    '''
                    slice0 = np.expand_dims(nct_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
                    if slice0.max() > 0:  # some of the image are totally black

                        # erase the end of registration which some of them are black
                        j =j +1
                        if j >=10: # 10 is the threshhold of bad image
                            target_path = target_path_nct
                            target_filename = 'regct_' + roots.split('/')[-1] + '_' + str(i) + IMAGE_TYPE
                            if not os.path.exists(target_path + '/' + target_filename):
                                scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
            except:
                print(file_path)



















































# import numpy as np
# import md
# import os
#
# Data_root_3d = '/data0/geyunhao/CT_MR/'
# Data_root_2d = '/home/geyunhao/Mapping/Mapping/pytorch-CycleGAN-and-pix2pix-master/datasets/MR2CT/'
# DATA_NAME = 'train'
# IMAGE_TYPE = '.mhd'
# '''
#     Turn the 3D slice to 2D slice and satisfy the structure of cycleGAnN
# '''
# # choose a templete from original MR & CT
# if os.path.exists(Data_root_3d + 'mr_ref.mhd') == False:
#     mr_3d_samp = md.read_image(Data_root_3d + '0001/MR.nii.gz') #  (x,y,z)
#     # mr_3d_samp_np = mr_3d_samp.to_numpy()# (z,y,x)
#     # mr_3d_samp.from_numpy(mr_3d_samp_np) # turn the np 2 image, the (z,y,x)auto 2 (x,y,z)
#     mr_slice = md.image3d_tools.center_crop(mr_3d_samp, [104, 70, 213], [2.4, 2.4, 2.4], [208, 140, 1])
#     md.write_image(mr_slice, Data_root_3d + 'mr_ref.mhd')
#
# if os.path.exists(Data_root_3d + 'nct_ref.mhd') == False:
#     nct_3d_samp = md.read_image(Data_root_3d + '0001/nfCT.nii.gz') #  (x,y,z)
#     nct_slice = md.image3d_tools.center_crop(nct_3d_samp, [104, 70, 213], [2.4, 2.4, 2.4], [208, 140, 1])
#     md.write_image(nct_slice, Data_root_3d + 'nct_ref.mhd')
#
# # get the 2d slice of MR
# mr_ref = md.read_image(Data_root_3d + 'mr_ref.mhd')
# nct_ref = md.read_image(Data_root_3d + 'nct_ref.mhd')
#
# target_path_mr = Data_root_2d + DATA_NAME + 'A'
# target_path_nct = Data_root_2d + DATA_NAME + 'B'
# if not os.path.exists(target_path_mr):
#     os.makedirs(target_path_mr)
# if not os.path.exists(target_path_nct):
#     os.makedirs(target_path_nct)
#
#
# for roots, dirs, files in os.walk(Data_root_3d + DATA_NAME):
#     for file in files:
#         file_path = os.path.join(roots, file)
#         if 'MR' in file:
#             mr_3d = md.read_image(file_path)  # (x,y,z)
#             mr_3d_np = mr_3d.to_numpy() # (z,y,x)
#             for i in range(mr_3d_np.shape[0]): # z
#                 slice0 = np.expand_dims(mr_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
#                 mr_ref.from_numpy(slice0)  # put slice data into ref
#                 target_path = target_path_mr
#                 target_filename = 'mr_' + roots.split('/')[-1] + '_'+ str(i) + IMAGE_TYPE
#                 md.write_image(mr_ref, target_path + '/' + target_filename)  #
#
#         elif 'nfCT' in file:
#             nct_3d = md.read_image(file_path)  # (x,y,z)
#             nct_3d_np = nct_3d.to_numpy()  # (z,y,x)
#             for i in range(nct_3d_np.shape[0]):  # z
#                 slice0 = np.expand_dims(nct_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
#                 nct_ref.from_numpy(slice0)  # put slice data into ref
#                 target_path = target_path_nct
#                 target_filename = 'nct_' + roots.split('/')[-1] + '_' + str(i) + IMAGE_TYPE
#                 md.write_image(mr_ref, target_path + '/' + target_filename)  #







# ipython code
# mr_3d_samp = md.read_image(Data_root_3d + '0001/MR.nii.gz') #  (x,y,z)
# mr_ref = md.read_image(Data_root_2d + 'mr_ref.mhd')
# mr_3d_samp_np = mr_3d_samp.to_numpy()
# slice0 = np.expand_dims(mr_3d_samp_np[0, :, :], 0) # get slice data without decay
# mr_ref.from_numpy(slice0)# put slice data into ref
# md.write_image(mr_ref, Data_root_2d + 'single-slice.mhd') #
# ex = md.read_image(Data_root_2d + 'single-slice.mhd')











# nct_3d_samp = md.read_image('/data0/geyunhao/CT_MR/0001/nfCT.nii.gz')
# ct_3d_samp = md.read_image('/data0/geyunhao/CT_MR/0001/CT.nii.gz')
