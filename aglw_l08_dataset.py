import os
import torch
from torch.utils.data import Dataset, DataLoader

from torchgeo.datasets import GeoDataset, RasterDataset, AbovegroundLiveWoodyBiomassDensity, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.datasets.utils import BoundingBox

import numpy as np
from scipy.ndimage import zoom        

from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
#import gdal
#gdal.PushErrorHandler('CPLQuietErrorHandler')

class BDLandsat(RasterDataset):
    filename_glob  = '*/*.tif'
    filename_regex = r'^.{5}_(?P<date>\d{4}-\d{2}-\d{2})'
    date_format    = '%Y-%m-%d'
    is_image       = True
    separate_files = True
    all_bands = ('B03', 'B04', 'B05','B06')
    rgb_bands = ('B03', 'B04', 'B05')

class BDSentinel(RasterDataset):
    filename_glob  = '*/*.tif'
    filename_regex = r'^.{5}_(?P<date>\d{4}\d{2}\d{2}T\d{6})'
    date_format    = '%Y%m%dT%H%M%S'
    is_image       = True
    separate_files = False
    all_bands = ('B02', 'B03', 'B04', 'B08', 'B11','QA60')
    rgb_bands = ('B03', 'B04', 'B08')

minx = 88.0844222351  + 1.0
maxx = 92.6727209818  - 1.0
miny = 20.670883287   + 1.0
maxy = 26.4465255803  - 1.0

#minx, maxx, miny, maxy = 90.45173571, 90.45837391, 23.00388088, 23.00640725
#minx, maxx, miny, maxy = 90.00457313 , 90.51279452, 23.52214718, 24.0443497
#root = '/media/irfan/TRANSCEND/satellite_data/bd_l08_cc5_b345' #bd_landsat_b2345_r200/2013-07-12'
#root         = '/media/irfan/TRANSCEND/satellite_data/bd_l8_5_150_B3B4B5B6/train'

root         = '/media/irfan/TRANSCEND/satellite_data/us_cpk_train_l8_5_150_B3B4B5B6/'
img_dataset  = BDLandsat(root)
#root = '/media/irfan/TRANSCEND/satellite_data/bd_s2_3_100_B2B3B4B8B11QA60/'
#img_dataset  = BDSentinel(root)

mask = 'cpk'
if mask == 'agb':
    mask_dataset = AbovegroundLiveWoodyBiomassDensity(paths='/media/irfan/TRANSCEND/satellite_data/aglw_cd', 
                                                      crs=None,
                                                      res=None,
                                                      transforms =None,
                                                      download=False,
                                                      cache=True)
if mask == 'cpk':
    #chesapeake_root = '/media/irfan/TRANSCEND/satellite_data/chesapeake'
    chesapeake_train = '/media/irfan/TRANSCEND/satellite_data/chesapeake_dwn/train/'
    chesapeake_val   = '/media/irfan/TRANSCEND/satellite_data/chesapeake_dwn/val/'
    
    class CPK(RasterDataset):
        filename_glob  = '*/*.tif'
        filename_regex = r'^.{2}_lc_(?P<date>\d{4})_2022-Edition'
        date_format    = '%Y'
        is_image       = False
        cmap = {
         0: (0, 0, 0, 0),
         1: (255, 0, 0, 255),
         2: (0, 255, 0, 255),
         3: (0, 0, 255, 255),
         4: (0, 0, 128, 255),
         5: (0, 128, 0, 255),
         6: (128, 0, 0, 255),
         7: (0, 197, 255, 255),
         8: (38, 115, 0, 255),
         9: (163, 255, 115, 255),
         10: (255, 170, 0, 255),
         11: (156, 156, 156, 255),
         12: (128, 128, 128, 255)}
    train_mask_dataset  = CPK(chesapeake_train, crs=img_dataset.crs, res=img_dataset.res)
    val_mask_dataset    = CPK(chesapeake_val, crs=img_dataset.crs, res=img_dataset.res)
    
train_dataset = img_dataset & train_mask_dataset
val_dataset   = img_dataset & val_mask_dataset


class NDVIDataset(Dataset):
    def __init__(self, split = 'train'):
        if split == 'train':
            self.dataset = train_dataset
            self.index   = train_dataset.index
            self.res     = train_dataset.res
        if split == 'val':
            self.dataset = val_dataset
            self.index   = val_dataset.index
            self.res     = val_dataset.res
        self.count   = 0
        
    def process_mask_mmseg(self,img, gt):
        res     = {}
        res['inputs']       = img
        #gt                  = np.uint8(gt/100)
        res['gt_sem_seg']   = gt 
        res['data_samples'] = SegDataSample()
        res['data_samples'].set_metainfo(dict(img_path=self.count, seg_map_path='', ori_shape=img.shape[1:],
                            img_shape=img.shape[1:]))
        #, pad_shape=[], scale_factor=[], flip= False,flip_direction=None, reduce_zero_label=None))
        res['data_samples'].gt_sem_seg = PixelData(data = torch.tensor(gt,dtype=torch.int64))
        return res
        
    def __getitem__(self,bbox):
        res     = self.dataset.__getitem__(bbox)
        img     = res['image'][:4].numpy().copy() # G, ,R ,N, S
        #nodata_ind = img[]
        img     = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        g, r, n, s = img
        gt      = res['mask'][0]#.numpy().copy()
        
        #g       = zoom(g,0.25)
        #r       = zoom(r,0.25)
        #n       = zoom(n,0.25)
        #gt      = zoom(res['mask'][0].numpy().copy(),0.25)
        ndvi    = np.clip( ((n - r) / ((n + r) + 1e-3)), -0.5, 0.5) + 0.5
        ndwi    = np.clip( ((g - n) / ((g + n) + 1e-3)), -0.5, 0.5) + 0.5
        ndbi    = np.clip( ((s - n) / ((s + n) + 1e-3)), -0.5, 0.5) + 0.5
        h,w     = ndvi.shape
        out     = np.concatenate([ndbi[None,:,:],ndvi[None,:,:],ndwi[None,:,:]],axis = 0)
        img     = torch.tensor(out,dtype=torch.float32)
        ### chesapeak 
        #gt                  = gt - 10
        gt[gt < 0]          = 0
        gt[gt > 12]         = 0
        res['mask']         = gt
        res['image']        = img
        ###
        res = self.process_mask_mmseg(img, gt)
        self.count += 1
        return res
        
train_dataset    = NDVIDataset(split = 'train')
val_dataset    = NDVIDataset(split = 'val')
'''
#dataset.__getitem__(0)
minx = 88.0844222351  + 1.0
maxx = 92.6727209818  - 1.0
miny = 20.670883287   + 1.0
maxy = 26.4465255803  - 1.0
bbox = BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=comb_dataset.bounds.mint, maxt=comb_dataset.bounds.maxt)
'''
train_sampler   = RandomGeoSampler(train_dataset, size=64) #, length=4)
val_sampler     = GridGeoSampler(val_dataset, 64, 64, units=Units.PIXELS)
#sampler        = RandomGeoSampler(ndvi_dataset, size=64) #, length=4)

def collate_fn(batch):
    nbatch = []
    for elem in batch:
        if torch.sum(elem['gt_sem_seg']>0) > 3600 and torch.sum(elem['inputs']==0.5) < 100:
            nbatch.append(elem)
    return stack_samples(nbatch)
'''
class CustomDataLoader(DataLoader):
    super().__iter__().__next__ = cls.next
    def next(self):
        ret = super().__iter__().__next__()
        if not len(ret['samples']):
            return self.next()
        else:
            return ret
'''

class CustomDataLoader():
    def __init__(self,dataset, sampler = None, batch_size = 1):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.sampler      = sampler
        self.sampler_iter = iter(sampler)
        self.count        = 0
        self.len          = len(sampler)
        self.dataset.len = self.__len__()
    def reset(self):
        self.count = 0
        self.sampler_iter = iter(self.sampler)
        #pass
    def __next__(self):
        nbatch = []
        while 1:
            if self.sampler_iter is None:
                self.reset()
            bbox = next(self.sampler_iter)
            data = self.dataset.__getitem__(bbox)
            if torch.sum(data['gt_sem_seg']>0) > 3600 and torch.sum(data['inputs']==0.5) < 100:
               nbatch.append(data)
            self.count += 1
            if len(nbatch) == self.batch_size:
                return stack_samples(nbatch)
            
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.sampler)//self.batch_size
            
TrainDataloader = CustomDataLoader(train_dataset, sampler=train_sampler, batch_size = 16)#, collate_fn=collate_fn, num_workers =0)
ValDataloader   = CustomDataLoader(val_dataset, sampler=val_sampler, batch_size = 16)#, collate_fn=collate_fn, num_workers =0)
        