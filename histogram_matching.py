import numpy as np
import cv2

asli = cv2.imread('asli1.jpg')

equ = np.copy(asli)

for i  in range (3) :
    equ[:,:,i] = cv2.equalizeHist(asli[:,:,i])

referensi =  cv2.imread('referensi1.jpg')
hasilmatch = np.copy(asli)

def hist_match(asli, referensi):

    asli_shape = asli.shape
    
    asli = asli.ravel()
    referensi = referensi.ravel()

    o_values, bin_idx, o_counts = np.unique(asli, return_inverse=True,return_counts=True)
    b_values, b_counts = np.unique(referensi, return_counts=True)

    o_quantiles = np.cumsum(o_counts).astype(np.float64)
    o_quantiles /= o_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    interp_t_values = np.interp(o_quantiles, b_quantiles, b_values)

    return interp_t_values[bin_idx].reshape(asli_shape)

for i  in range (3) :
    hasilmatch[:,:,i] = hist_match(asli[:,:,i], referensi[:,:,i])

cv2.imwrite('hasil1.jpg', hasilmatch)