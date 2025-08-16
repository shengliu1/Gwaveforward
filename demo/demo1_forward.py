# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:29:09 2022
D:\MyResearch\abic_inversion\freqinv

Gravity anomaly forward by two ways
@author: chens
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
# local imports
import freqinv.su as su


## 1. generate a model 
def gen_grav(shape):
   
    # X、Y、Z，North/East/Depth model 1
    model = [ su.Prism( 2000,  4000, -3000,  1000, 400, 800, {'density': 1.5}),
              su.Prism(-2000,     0, -1000,  2000, 400, 800, {'density': -1.0}),
              su.Prism(-4000, -1000, -4000, -3000, 300, 700, {'density': 1.0}),
              su.Prism( 1000,  4000,  3000,  4000, 300, 700, {'density': -1.5})]
    
    
    # X、Y、Z，North/East/Depth model 2
    #model = [su.Prism( -500,   500, -2700, -2500,  100,  400, {'density': 1.0}),
    #         su.Prism(-1000,  1000, -3000, -2000,  400,  700, {'density': 1.0}),
    #         su.Prism(-1500,  1500, -3500, -1500,  700, 1000, {'density': 1.0}),
    #         su.Prism(-2000,  2000, -4000, -1000, 1000, 1300, {'density': 1.0}),
    #         su.Prism(-2500,  2500, -4500,  -500, 1300, 1600, {'density': 1.0}),
    #         su.Prism(-3000,  3000, -5000,     0, 1600, 1900, {'density': 1.0}),
    #         su.Prism( -500,   500,  2500,  2700,  100,  400, {'density': -1.0}),
    #         su.Prism(-1000,  1000,  2000,  3000,  400,  700, {'density': -1.0}),
    #         su.Prism(-1500,  1500,  1500,  3500,  700, 1000, {'density': -1.0}),
    #         su.Prism(-2000,  2000,  1000,  4000, 1000, 1300, {'density': -1.0}),
    #         su.Prism(-2500,  2500,   500,  4500, 1300, 1600, {'density': -1.0}),
    #         su.Prism(-3000,  3000,     0,  5000, 1600, 1900, {'density': -1.0}),]


    xp, yp, zp = su.gridderRegular((-5000, 5000, -5000, 5000), shape, z=10.0)
	
    
	#the space domain approach (Nagy et al., 2000, 2002)
    field0 = su.gz(xp, yp, zp, model)
    field0x = su.gx(xp, yp, zp, model)
    field0y = su.gy(xp, yp, zp, model)
    gxx = su.gxx(xp, yp, zp, model)
    gxy = su.gxy(xp, yp, zp, model)
    gxz = su.gxz(xp, yp, zp, model)
    gyy = su.gyy(xp, yp, zp, model)
    gyz = su.gyz(xp, yp, zp, model)
    gzz = su.gzz(xp, yp, zp, model)

    
	#the wavenumber domain approach (this study)
    fieldf = su.gzfreq(xp, yp, zp, shape, model)
    fieldfx = su.gxfreq(xp, yp, zp, shape, model)
    fieldfy = su.gyfreq(xp, yp, zp, shape, model)
    gxxf   = su.gxxfreq(xp, yp, zp, shape, model)
    gxyf   = su.gxyfreq(xp, yp, zp, shape, model)
    gxzf   = su.gxzfreq(xp, yp, zp, shape, model)
    gyyf   = su.gyyfreq(xp, yp, zp, shape, model)
    gyzf   = su.gyzfreq(xp, yp, zp, shape, model)
    gzzf   = su.gzzfreq(xp, yp, zp, shape, model)
    
    
    #field1 = giutils.contaminate(field0, 0.05, percent = True)
    return field0, fieldf, field0x, fieldfx, field0y, fieldfy, gxx, gxxf, gxy, gxyf, gxz, gxzf, gyy, gyyf, gyz, gyzf, gzz, gzzf, xp, yp, zp


if __name__ == '__main__':

    print('Hello freqinv! Gravity Forward!')
   
    shape = (10,  10)
    shape = (20,  20)
    shape = (50,  50)
    shape = (100, 100)
    shape = (300, 300)
    
    gu, gulist, gux, gulistx, guy, gulisty, gxx, gxxf, gxy, gxyf, gxz, gxzf, gyy, gyyf, gyz, gyzf, gzz, gzzf, xp, yp, zp = gen_grav(shape)
    print(np.linalg.norm(gu-gulist)/np.linalg.norm(gu))


    #plt.subplot(3, 3, 7)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gravity anomlay forward by space-domain method')
    levels = su.contourf(yp * 0.001, xp * 0.001, gu, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gu, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gz by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gulist, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, gulist, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gz_space-gz_freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gu-gulist, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, gu-gulist, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()
    
    
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gx anomlay forward by space-domain method')
    levels = su.contourf(yp * 0.001, xp * 0.001, gux, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gux, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gx by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gulistx, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, gulistx, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gx_space-gx_freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gux-gulistx, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, gux-gulistx, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gy anomlay forward by space-domain method')
    levels = su.contourf(yp * 0.001, xp * 0.001, guy, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, guy, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gy by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gulisty, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, gulisty, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gy_space-gy_freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, guy-gulisty, shape, 15)
    cb = plt.colorbar( )
    su.contour(yp * 0.001, xp * 0.001, guy-gulisty, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    #plt.subplot(3, 3, 1)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxx by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxx, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxx, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxx by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxxf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxxf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    #plt.subplot(3, 3, 2)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxy by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxy, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxy, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxy by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxyf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxyf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()
    

    #plt.subplot(3, 3, 3)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxz by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxz, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxz, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gxz by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gxzf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gxzf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    #plt.subplot(3, 3, 5)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gyy by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gyy, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gyy, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gyy by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gyyf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gyyf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()


    #plt.subplot(3, 3, 6)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gyz by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gyz, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gyz, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gyz by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gyzf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gyzf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()
    

    #plt.subplot(3, 3, 9)
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gzz by space')
    levels = su.contourf(yp * 0.001, xp * 0.001, gzz, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gzz, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()
    
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    plt.title('gzz by freq')
    levels = su.contourf(yp * 0.001, xp * 0.001, gzzf, shape, 15)
    cb = plt.colorbar()
    su.contour(yp * 0.001, xp * 0.001, gzzf, shape,
                levels, clabel=False, linewidth=0.1)
    plt.show()
    
    
    
    df1 = pd.DataFrame(columns=['x','y','z','gz','gzf','gz-gzf','gxx','gxxf','gxx-gxxf','gxy','gxyf','gxy-gxyf',
                                'gxz','gxzf','gxz-gxzf','gyy','gyyf','gyy-gyyf','gyz','gyzf','gyz-gyzf',
                                'gzz','gzzf','gzz-gzzf'])
    df1['x'] = yp
    df1['y'] = xp
    df1['z'] = zp
    df1['gz'] = gu
    df1['gzf'] = gulist
    df1['gz-gzf'] = gu-gulist
    
    df1['gxx'] = gxx
    df1['gxxf'] = gxxf
    df1['gxx-gxxf'] = gxx-gxxf
    
    df1['gxy'] = gxy
    df1['gxyf'] = gxyf
    df1['gxy-gxyf'] = gxy-gxyf
    
    df1['gxz'] = gxz
    df1['gxzf'] = gxzf
    df1['gxz-gxzf'] = gxz-gxzf
    
    df1['gyy'] = gyy
    df1['gyyf'] = gyyf
    df1['gyy-gyyf'] = gyy-gyyf
    
    df1['gyz'] = gyz
    df1['gyzf'] = gyzf
    df1['gyz-gyzf'] = gyz-gyzf
    
    df1['gzz'] = gzz
    df1['gzzf'] = gzzf
    df1['gzz-gzzf'] = gzz-gzzf
      
       
    #df1.to_csv('.\\G_tensor_freq_10.csv')
    #df1.to_csv('.\\G_tensor_freq_20.csv')
    #df1.to_csv('.\\G_tensor_freq_50.csv')
    #df1.to_csv('.\\G_tensor_freq_100.csv')
    #df1.to_csv('.\\G_tensor_freq_200.csv')
    
    
    #df1.to_csv('.\\G_tensor_freq_2_10.csv')
    #df1.to_csv('.\\G_tensor_freq_20.csv')
    #df1.to_csv('.\\G_tensor_freq_50.csv')
    #df1.to_csv('.\\G_tensor_freq_100.csv')
    #df1.to_csv('.\\G_tensor_freq_200.csv')
    #df1.to_csv('.\\G_tensor_freq_300.csv')
    #df1.to_csv('.\\G_tensor_freq_400.csv')
    
    
    
