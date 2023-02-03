from .imports import *

# from .expfile import *
# from .analysis import *
from .mathutil import *
# from .plotutil import *
# from .imagutil import *
# from .mako import *
# from .adam import *

def psf(x, w):
#     return 1/(np.pi*w/2)*np.exp(-2*(x**2)/w)
    # return np.sqrt(2)/np.sqrt(np.pi)/w*np.exp(-2*(x**2)/w**2)
    return np.exp(-2*(x**2)/w**2)

def box(x, R):
    return np.where(x>R, 0, 1)

def deconvolve(img, w, iters):
    a = round(w)
    x = np.arange(-a+1,a)
    y = np.arange(-a+1,a)

    xx, yy = np.meshgrid(x, y)
    psfM = psf(np.sqrt(xx**2+yy**2), w)

    img = img - img.min()
    norm = np.max(img)
    img = img/norm

    imgRL = restoration.richardson_lucy(img, psfM, iterations=iters)*norm
    return imgRL

def atomVal(img, mask, w = 6, iters = 20):
    return np.sum(mask*deconvolve(img, w, iters))

def getMasks(mimg, fftN = 2000, N = 10, wmask = 3, supersample = None, mode = 'gauss', FFT = True, peakParams = [10,10], output = True, coords = None, mindist=100, disttozero=[50,100,100],
             get_mask_centers = False):
    """Given an averaged atom image, returns list of masks, where each mask corresponds to the appropriate mask for a single atom."""

    if FFT:

        fimg = np.fft.fft2(mimg, s = (fftN,fftN))
        fimg = np.fft.fftshift(fimg)
        fimgAbs = np.abs(fimg)
        fimgArg = np.angle(fimg)

        # fimgMax = ndi.maximum_filter(fimg, size = 100, mode = 'constant')
        fMaxCoord = peak_local_max(fimgAbs, min_distance=mindist, threshold_rel=.3)
        print(len(fMaxCoord))
        # fMaxBool = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1, num_peaks = 4, indices=False)

        fMaxCoord = fMaxCoord[fMaxCoord[:,0]-fftN/2>disttozero[0]]# Restrict to positive quadrant
        fMaxCoord = fMaxCoord[fMaxCoord[:,1]-fftN/2>disttozero[1]] # Restrict to positive quadrant
        fMaxCoord = fMaxCoord[fMaxCoord.sum(axis=1)-fftN>disttozero[2]] # Restrict to positive quadrant

    #     xsort = np.lexsort((fMaxCoord[:,0]+fMaxCoord[:,1]))
    #     ysort = np.lexsort((fMaxCoord[:,1]+fMaxCoord[:,0]))

    #     ysort = np.argsort(fMaxCoord[:,1])
    #     xsort = np.argsort(fMaxCoord[:,0])

        xsort = np.argsort(fMaxCoord[:,1]+fMaxCoord[:,0]/5)
        ysort = np.argsort(fMaxCoord[:,0]+fMaxCoord[:,1]/5)

        xpeak, ypeak = fMaxCoord[xsort[0]], fMaxCoord[ysort[0]]

    #     print(fMaxCoord)
    #     print(xsort)
    #     print(ysort)
        if output == True:

            plt.imshow(fimgAbs)
            plt.colorbar()
            plt.plot(fMaxCoord[:,1], fMaxCoord[:,0],'g.')
            # plt.plot([xpeak[0], ypeak[0]],[xpeak[1],ypeak[1]],'r.')

            plt.plot([xpeak[1]],[xpeak[0]],'r.')
            plt.plot([ypeak[1]],[ypeak[0]],'b.')
            plt.vlines(fftN/2+disttozero[0],0, fftN, colors='b', linestyles='-')
            plt.hlines(fftN/2+disttozero[1], 0, fftN, colors='r', linestyles='-')
            # plt.plot(0,1000,'r.')

            plt.show()

        freqs = np.fft.fftfreq(fftN)
        freqs = np.fft.fftshift(freqs)
        fx, fy = freqs[xpeak], freqs[ypeak]
        dx, dy = 1/fx, 1/fy
        # dx = arr([dx[1]/dy[0], dx[1]])
        # dy = arr([dy[0],dy[0]/dy[1]])

        phix, phiy = fimgArg[xpeak[0], xpeak[1]], fimgArg[ypeak[0], ypeak[1]]

        # if phix<0:
        #     phix = -phix
        # if phiy<0:
        #     phiy = -phiy

        normX = np.sqrt(np.sum(fx**2))
        dx = (1/normX)*(fx/normX)

        normY = np.sqrt(np.sum(fy**2))
        dy = (1/normY)*(fy/normY)

        dx[1]=-dx[1]
        dy[0]=-dy[0]
        tmp = dy[0]
        dy[0] = dx[1]
        dx[1] = tmp

        if supersample!= None:
            dx, dy = dx/supersample, dy/supersample
            N = ((N-1)*supersample+1).astype(int)

        if type(N)==int:
            ns = np.arange(N)

            px = arr([(dx*ind) for ind in ns])
            py = arr([(dy*ind) for ind in ns])

            pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N**2,2))

        elif type(N)==list:
            nsx = np.arange(N[1])
            nsy = np.arange(N[0])

            px = arr([(dx*ind) for ind in nsx])
            py = arr([(dy*ind) for ind in nsy])

            pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N[0]*N[1] ,2))
        else:
            raise ValueError('Invalid array dimensions.')

    elif not FFT:
            filter_size = peakParams[0]
            threshold = peakParams[1]
            dmin=ndi.filters.minimum_filter(mimg,filter_size)
            dmax=ndi.filters.maximum_filter(mimg,filter_size)

            maxima = (mimg==dmax)

            diff = ((dmax-dmin)>threshold)
            maxima[diff == 0] = 0
            maxima[diff != 0] = 1 #Aruku added to elminate the double count error

            labeled, num_objects = ndi.label(maxima)
            pts = np.array(ndi.center_of_mass(mimg, labeled, range(1, num_objects+1)))
            # print(pts)
            sort = np.argsort(pts[:,1])
            pts = pts[sort]
#             pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N[0]*N[1] ,2))
    if output == True:
        plt.imshow(mimg)
        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:
            plt.plot(pts[:,1],pts[:,0],'r.')
        plt.show()

    x = np.arange(len(mimg[0]))
    y = np.arange(len(mimg[:,0]))

    xx, yy = np.meshgrid(x, y)

    if mode == 'gauss':
        masks = arr([psf(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
        if coords != None:
            masks = arr([psf(np.sqrt((xx-coords[1])**2+(yy-coords[0])**2), wmask) for i in range(len(pts))])
    if mode == 'box':
        masks = arr([box(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
    if output == True:
        plt.imshow(np.sum(masks, axis=0))
        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:
            plt.plot(pts[:,1],pts[:,0],'r.')
        plt.show()
        if FFT:
            print(dx, dy)

    if (get_mask_centers == True):
        return masks, pts
    else:
        return masks

# def getLatticeMasks(mimg, tweezerSpacing = 4, fftN = 2000, N = 10, wmask = 3, mode = 'gauss'):
#     """Given an averaged atom image, returns list of masks, where each mask corresponds to the appropriate mask for a single atom."""

#     fimg = np.fft.fft2(mimg, s = (fftN,fftN))
#     fimg = np.fft.fftshift(fimg)
#     fimgAbs = np.abs(fimg)
#     fimgArg = np.angle(fimg)

#     # fimgMax = ndi.maximum_filter(fimg, size = 100, mode = 'constant')
#     fMaxCoord = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1)
#     # fMaxBool = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1, num_peaks = 4, indices=False)

#     fMaxCoord = fMaxCoord[fMaxCoord[:,0]-fftN/2>-fftN/100] # Restrict to positive quadrant
#     fMaxCoord = fMaxCoord[fMaxCoord[:,1]-fftN/2>-fftN/100] # Restrict to positive quadrant
#     fMaxCoord = fMaxCoord[fMaxCoord.sum(axis=1)-fftN>fftN/100] # Restrict to positive quadrant

# #     xsort = np.lexsort((fMaxCoord[:,0]+fMaxCoord[:,1]))
# #     ysort = np.lexsort((fMaxCoord[:,1]+fMaxCoord[:,0]))

# #     ysort = np.argsort(fMaxCoord[:,1])
# #     xsort = np.argsort(fMaxCoord[:,0])

#     xsort = np.argsort(fMaxCoord[:,1]+fMaxCoord[:,0]/2)
#     ysort = np.argsort(fMaxCoord[:,0]+fMaxCoord[:,1]/2)

#     xpeak, ypeak = fMaxCoord[xsort[0]], fMaxCoord[ysort[0]]

# #     print(fMaxCoord)
# #     print(xsort)
# #     print(ysort)

#     plt.imshow(fimgAbs)
#     plt.plot(fMaxCoord[:,1], fMaxCoord[:,0],'g.')
#     # plt.plot([xpeak[0], ypeak[0]],[xpeak[1],ypeak[1]],'r.')
#     plt.plot([xpeak[1]],[xpeak[0]],'r.')
#     plt.plot([ypeak[1]],[ypeak[0]],'b.')
#     # plt.plot(0,1000,'r.')

#     plt.show()

#     freqs = np.fft.fftfreq(fftN)
#     freqs = np.fft.fftshift(freqs)
#     fx, fy = freqs[xpeak], freqs[ypeak]
#     dx, dy = 1/fx, 1/fy
#     # dx = arr([dx[1]/dy[0], dx[1]])
#     # dy = arr([dy[0],dy[0]/dy[1]])

#     phix, phiy = fimgArg[xpeak[0], xpeak[1]], fimgArg[ypeak[0], ypeak[1]]

#     # if phix<0:
#     #     phix = -phix
#     # if phiy<0:
#     #     phiy = -phiy

#     normX = np.sqrt(np.sum(fx**2))
#     dx = (1/normX)*(fx/normX)

#     normY = np.sqrt(np.sum(fy**2))
#     dy = (1/normY)*(fy/normY)

#     dx[1]=-dx[1]
#     dy[0]=-dy[0]
#     tmp = dy[0]
#     dy[0] = dx[1]
#     dx[1] = tmp

#     if N.size == 1:
#         ns = np.arange(N)

#         px = arr([(dx*ind) for ind in ns])
#         py = arr([(dy*ind) for ind in ns])

#         pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N**2,2))

#     elif N.size == 2:
#         nsx = np.arange(N[1])
#         nsy = np.arange(N[0])

#         px = arr([(dx*ind) for ind in nsx])
#         py = arr([(dy*ind) for ind in nsy])

#         pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N[0]*N[1] ,2))
#     else:
#         raise ValueError('Invalid array dimensions.')


#     for pt in pts:


#     plt.imshow(mimg)
#     plt.plot(pts[:,1],pts[:,0],'r.')
#     plt.show()

#     x = np.arange(len(mimg[0]))
#     y = np.arange(len(mimg[:,0]))

#     xx, yy = np.meshgrid(x, y)

#     if mode == 'gauss':
#         masks = arr([psf(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
#     if mode == 'box':
#         masks = arr([box(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
#     plt.imshow(np.sum(masks, axis=0))
#     plt.plot(pts[:,1],pts[:,0],'r.')
#     plt.show()

#     print(dx, dy)

#     dxLat, dyLat = dx/tweezerSpacing, dy/tweezerSpacing
#     N = ((arr(N)-1)*tweezerSpacing+1).astype(int)

#     return masks

def crop(img, sigma = 2, offset = 100, plots = False, coords = False, pad = 0):
    """Crop image to bright region. Returns cropped image or, if coords set to True, cropped image and coordinates [xmin, xmax, ymin, ymax]."""
    imgLP = sp.ndimage.gaussian_filter(img, sigma)
    mval = np.mean(imgLP)
    xbin = np.mean(imgLP, axis = 1)-mval/offset
    ybin = np.mean(imgLP, axis = 0)-mval/offset
    xCross = np.where(np.diff(np.sign(xbin)))[0]
    yCross = np.where(np.diff(np.sign(ybin)))[0]

    if plots:
        plt.plot(xbin)
        plt.plot(ybin)
        plt.show()

    if coords:
        return img[xCross[0]-pad:xCross[1]+pad, yCross[0]-pad:yCross[1]+pad], arr([xCross[0]-pad, xCross[1]+pad, yCross[0]-pad, yCross[1]+pad])
    else:
        return img[xCross[0]-pad:xCross[1]+pad, yCross[0]-pad:yCross[1]+pad]
