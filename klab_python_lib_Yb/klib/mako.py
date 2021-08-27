from .imports import *

# from .expfile import *
# from .analysis import *
from .mathutil import *
# from .plotutil import *
# from .imagutil import *
# from .mako import *
# from .adam import *

# photons per pixel count ad gain = 40 dB
mako2Dcal = 0.61 #
mako3Dcal = 3.638 #(1.38 was for the old imaging system) # SN 536973476

def loadBMPs(path, nameFormat):
    names = arr(os.listdir(path))
    print(names)
    inds = arr([int((i.replace(nameFormat,"")).replace(".bmp","")) for i in names])
    sort = np.argsort(inds)
    names = names[sort]
    imgs = []
    for name in names:
        img  = arr(Image.open(path+name))
        imgs.append(img)

    return imgs

def sortImgs(exp, imgs):
    """Sort list of imported images (imgs) by hdf5 experiment object (exp) returned from Chimera."""
    key = exp.key
    reps = exp.reps
    sort = np.argsort(key)
    shape = imgs.shape
    imgs = np.reshape(imgs, (len(key), reps, shape[-2], shape[-1]))
    return key[sort], imgs[sort]

def GetMakoPhase(fp,rotangle=None,sigma=3,plot=False):
    """Returns phase of fringes, fit params, and perr in image specified by filepath (fp) with respect to center of Gaussian. Set rotational angle (rotangle) to make fringe kvector vertical. Set sigma to be just longer than the period (in pixels) of the fringes. Set plot=True to display intermediate plots for debugging."""
    if rotangle is None: ### Rotation angle not given ###
        file = Image.open(fp)
        img = np.array(file)

        ### Coarse cropping ### (runs fast and makes the Gaussian later much faster)
        img_lpf = scipy.ndimage.gaussian_filter(img,20)
        img_pkind = np.unravel_index(np.argmax(img_lpf, axis=None), img_lpf.shape)
        dx = 150; dy=dx;
        mimg = img[img_pkind[0]-dy:img_pkind[0]+dy,img_pkind[1]-dx:img_pkind[1]+dx]
        if plot:
            plt.imshow(img_lpf)
            print(img_pkind)
            plt.plot(img_pkind[1],img_pkind[0],'r.')
            plt.show()
            plt.imshow(mimg)
            plt.show()

        ### Isolate fringe signal in image###
        vgfimg = scipy.ndimage.gaussian_filter1d(mimg,sigma,axis=0) #vertical gaussian filter
        hpfimg = mimg-vgfimg

        if plot:
            plt.imshow(vgfimg)
            plt.show()
            plt.imshow(hpfimg)
            plt.show()

        ### Isolate fringe orientation and spacing ###
        fftN = 5000
        w=289 #This is calculated for real when rotangle is specified, but is slow to do so. This is a guess based on previously obtained values of w.
        fimg = np.fft.fft2(hpfimg, s=(fftN,fftN))
        fimg = np.fft.fftshift(fimg)
        fimgAbs = np.abs(fimg)
        fimgArg = np.angle(fimg)
        fMaxCoord = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1 , exclude_border=False)
        # fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,0]-fftN/2)>fftN/2-w/10] # exclude center
        fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,0]-fftN/2)>w] # exclude center
        fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,1]-fftN/2)<w] # exclude center
        if plot:
            plt.imshow(np.log(fimgAbs))
            for peak in fMaxCoord:
                plt.plot([peak[1]],[peak[0]],'r.')
            plt.show()

        ### Unrotate image and extract fringe signal, fit to cosine ###
        rotang = -90+np.arctan2(fMaxCoord[0,0]-fMaxCoord[1,0],fMaxCoord[0,1]-fMaxCoord[1,1])*180/np.pi
        print('Rotation angle not provided, calculated to be: ',str(rotang),' deg')
        return GetMakoPhase(fp,rotang, sigma, plot)
    else: ### Rotation angle given ###
        file = Image.open(fp)
        img = np.array(file)

        ### Coarse cropping, rotate image ### (runs fast and makes the Gaussian later much faster)
        img_lpf = scipy.ndimage.gaussian_filter(img,20)
        img_pkind = np.unravel_index(np.argmax(img_lpf, axis=None), img_lpf.shape)
        if plot:
            print('Coarse Crop Center: ', img_pkind)
        dx = 250; dy=dx;
        cimg = img[img_pkind[0]-dy:img_pkind[0]+dy,img_pkind[1]-dx:img_pkind[1]+dx]
        mimg = scipy.ndimage.rotate(cimg, rotangle)
        clipsides = np.max([np.shape(mimg)[0]-np.shape(cimg)[0], np.shape(mimg)[1]-np.shape(cimg)[1]])
        mimg = mimg[clipsides:-clipsides,clipsides:-clipsides]
        if plot:
            plt.imshow(img_lpf)
            plt.plot(img_pkind[1],img_pkind[0],'r.')
            plt.show()
            plt.imshow(mimg)
            plt.show()

        ### Gaussian fit for precise centering ###
        img_lpf = scipy.ndimage.gaussian_filter(img_lpf[img_pkind[0]-dy:img_pkind[0]+dy,img_pkind[1]-dx:img_pkind[1]+dx],25)
        if plot:
            plt.imshow(img_lpf)
            plt.show()
        gfit, gpopt = gaussianBeamFit(img_lpf);
        x0 = gpopt[1]; y0 = gpopt[2];
        w = int(2*gpopt[3]);
        if plot:
            print('Fitted Gaussian Coords, Width: ',np.round(x0,3),np.round(y0,3),w)

        ### Isolate fringe signal in image ###
        vgfimg = scipy.ndimage.gaussian_filter1d(mimg,sigma,axis=0) #vertical gaussian filter
        hpfimg = mimg-vgfimg
        if plot:
            plt.imshow(vgfimg)
            plt.show()
            plt.imshow(hpfimg)
            plt.show()
            print('hpf image size: ',np.shape(hpfimg))

        ### Isolate fringe orientation and spacing ###
        fftN = 5000
        fimg = np.fft.fft2(hpfimg, s=(fftN,fftN))
        fimg = np.fft.fftshift(fimg)
        fimgAbs = np.abs(fimg)
        fimgArg = np.angle(fimg)
        fMaxCoord = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1 , exclude_border=False)
        # fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,0]-fftN/2)>fftN/2-w/10] # exclude center
        fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,0]-fftN/2)>w] # exclude center
        fMaxCoord = fMaxCoord[np.abs(fMaxCoord[:,1]-fftN/2)<w] # exclude center
        if plot:
            plt.imshow(np.log(fimgAbs))
            for peak in fMaxCoord:
                plt.plot([peak[1]],[peak[0]],'r.')
            plt.show()

        ### Extract fringe signal, fit to cosine ###
        #print(x0, y0, w)
        fringes = np.mean(hpfimg[int(y0)-50:int(y0)+50,int(x0)-50:int(x0)+50],axis=1)
        xs = (range(np.shape(hpfimg)[0])-y0)[int(y0)-50:int(y0)+50] #Reference fringes to center of Gaussian fit, in units of the waist size.
        xdense = np.linspace(xs[0],xs[-1],500)
        f_px = np.linalg.norm(((fMaxCoord[0,0]-fMaxCoord[1,0])/np.shape(fimg)[0]/2.0,(fMaxCoord[0,1]-fMaxCoord[1,1])/np.shape(fimg)[0]/2.0)) #frequency in pixels from FFT
        f = f_px #*w #NOT frequency in units of the waist size.
        params, perr = cosFitF(xs, fringes, f, ic = [np.max(fringes)-np.mean(fringes),0,np.mean(fringes)], bd = ([0, -1.5*np.pi, 0],[1000, 1.5*np.pi, 1000]))
        phase = np.round(params[1]*180/np.pi,2)

        if plot:
            fringe_fig = plt.figure(figsize = (12,4))
            plt.plot(xs,fringes,'ko-')
            plt.plot(xdense,params[0]*np.cos(xdense*2.0*np.pi*f+params[1])+params[2],'r-')
            plt.title('phase = '+str(phase))
            fringe_fig.show()
            return phase, params, perr, fringe_fig
        else:
            return phase, params, perr
