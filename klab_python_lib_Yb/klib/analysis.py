from .imports import *

from .expfile import *
# from .analysis import *
from .mathutil import *
# from .plotutil import *
from .imagutil import *

import klib.experiment_constants as exc

# from .mako import *
# from .adam import *

### Main data object
class parsedData:
    """Wrapper for parsing HDF5 experiment file. Access original file with "raw" attribute, other attributes include keyName, keysort, fillfracs, points, rois, and roisums."""
    def __init__(self, dataAddress, fnum, countsperphoton = 70, thresh = 20, plots = True, mode = 'mask', masks = 0, w = 0, iters = 0, crop = 0):
        """Load in data file fnum from folder dataAddress. If setting ROI automatically, roi settings order: roi_size, filter_size, threshold. Otherwise provide full list of rois in roiSettings. Returns keySort, array of atom counts with dimensions [variation number, image number in experiment]"""

        exp = ExpFile(dataAddress+'Raw Data\\', fnum)
        keyName, keyVals, nreps, images = exp.key_name, exp.key, exp.reps, exp.pics
        variations = len(keyVals)
    #     print(keyVals)

        # Subtract mean of first 10 rows from all images, helps get rid of noise near CCD readout edge.
        meanimg = np.mean(images, axis=0)
        wx, wy = meanimg.shape

        bg = np.mean(meanimg[0:7,0:7])
        meanimg = np.subtract(meanimg, bg)
        # bgrow = np.mean(meanimg[0:7],axis=0)
        # meanimg = arr([i-bgrow for i in meanimg])
        # bgcol = np.mean(meanimg[:,0:7],axis=1)
        # meanimg = meanimg.transpose()
        # plt.imshow(meanimg)
        # meanimg = arr([i - bgcol for i in meanimg])
        # meanimg = meanimg.transpose()

        # bg_img = np.tile(bgrow, (wx,1))+np.tile(bgcol,(wy,1)).transpose()
        print(images.shape[0])
        for i in range(images.shape[0]):
            img = images[i]
            img = img - bg
            images[i]=img
        images=arr(images)

        if mode == 'decon' or mode == 'mask':
#             xmin = arr(rois)[:,0].min()
#             xmax = arr(rois)[:,1].max()
#             ymin = arr(rois)[:,2].min()
#             ymax = arr(rois)[:,3].max()
            [xmin, xmax, ymin, ymax] = crop

            if plots:
                masksum = np.sum(masks, axis=0)
#                 masksum = np.pad(masksum, ((ymin-pad, 0), (xmin-pad, 0)), mode = 'constant', constant_values = 0)
                plt.imshow(meanimg[xmin:xmax, ymin:ymax])
                plt.contour(masksum, 1, colors = 'r', alpha = 0.5)
                plt.show()

        if mode == 'decon':
            images_crop = images[:, ymin-pad:ymax+pad, xmin-pad:xmax+pad]
            images_rl = list(map(lambda image: deconvolve(image, w, iters), images_crop))
            roisums = np.array(list(map(lambda image:
                                        list(map(lambda mask:
                                                 np.sum(mask*image)-bglevel,
                                                 masks)),
                                        images_rl)))
        elif mode == 'box':
        # represent the roi sums as a 3-dimensional array.  Axes are variations, trials, rois.
            roisums = np.array(list(map(lambda image:
                                        list(map(lambda roi:
                                                 get_roi_sum(image, roi, bgoff, display=False),
                                                 rois)),
                                        images)))
        elif mode == 'mask':
#             xmin = arr(rois)[:,0].min()
#             xmax = arr(rois)[:,1].max()
#             ymin = arr(rois)[:,2].min()
#             ymax = arr(rois)[:,3].max()

            [xmin, xmax, ymin, ymax] = crop

#             images_crop = images[:, ymin-pad:ymax+pad, xmin-pad:xmax+pad]
            images_crop = images[:, xmin:xmax, ymin:ymax]
            roisums = np.array(list(map(lambda image:
                                        list(map(lambda mask:
                                                 np.sum(mask*image),
                                                 masks)),
                                        images_crop)))
        else:
            raise ValueError('Invalid ROI mode. Available modes are box, mask, and decon.')

        # Nice way of sorting in multiple dimensions. TODO: better handling for arbitrary keyVal shapes.
        if len(keyVals.shape)==1:
            sort = np.argsort(keyVals)
#             sort = arr(range(len(keyVals)))
        elif len(keyVals.shape)==2:
            sort = np.lexsort((keyVals[:,0],keyVals[:,1]))

        keySort = keyVals[sort]

        npics = (images.shape[0]//nreps)//keyVals.shape[0]
        images = images.reshape((variations, nreps, npics, images.shape[1], images.shape[2]))
        if mode == 'decon' or mode == 'mask':
            roisums = roisums.reshape(variations, nreps, npics, len(masks))
            self.roisums_old = roisums.reshape(variations, nreps*npics, len(masks))
        elif mode == 'box':
            roisums = roisums.reshape(variations, nreps, npics, len(rois))
            self.roisums_old = roisums.reshape(variations, nreps*npics, len(rois))
        #     roisums = roisums.reshape(variations, images.shape[0]//variations, len(rois))

        imsort = images[sort]
        roisums = roisums[sort]

        atom_thresh = thresh*countsperphoton
    #   Binarize roisums and average over reps. Axes are variation, image number in sequence, rois
        binarized = np.clip(roisums, atom_thresh, atom_thresh+1) - atom_thresh
        fill_first = np.mean(binarized[:,:,0,:], axis = (1,2))
        fill_last = np.mean(binarized[:,:,-1,:], axis = (1,2))
        fill_rois = np.mean(binarized[:,:,0,:], axis = 1)
        fillfracs = np.mean(binarized, axis = (1,2,3))
    #     print(tuple(np.arange(1, binarized.ndim)))

        points = len(masks)*nreps
        z = 1.96 # corresponds to 95% CI

        error_first = z*np.sqrt(fill_first*(1-fill_first)/points)
        error_last = z*np.sqrt(fill_last*(1-fill_last)/points)
        error = z*np.sqrt(fillfracs*(1-fillfracs)/points)

        makedirifneed(dataAddress+'losses/')
        makedirifneed(dataAddress+'fileparams/')
        f = open(dataAddress+'fileparams/'+str(fnum)+'.pkl','wb')
        pickle.dump(exp.return_variables(),f)
        f.close()


    #     lowerfrac = z*np.sqrt(fillfracs*(1-fillfracs)/points)
        self.data_address = dataAddress
        self.error = error
        self.error_first = error_first
        self.error_last = error_last

        self.imsort = imsort

#         roiimgs = []
#         for roi in rois:
#             rimg = meanimg[roi[2]:roi[3],roi[0]:roi[1]]
#             roiimgs.append(rimg)
#         self.roiimgs = arr(roiimgs)

#         roi = rois[len(rois)//2]
#         self.roiimg = meanimg[roi[2]:roi[3],roi[0]:roi[1]]

        self.fillfracs = fillfracs
        self.fill_first = fill_first
        self.fill_last = fill_last
        self.fill_rois = fill_rois
        self.raw = exp
        self.keyName = keyName
        self.keySort = keySort
        self.points = points
#         self.rois = rois
        self.roisums = roisums
        self.fnum = fnum
        self.thresh = thresh
        self.countsperphoton = countsperphoton
        self.binarized = binarized
        self.npics = npics
        self.nreps = nreps
        self.meanimg = meanimg

        self.masks = masks

def makedirifneed(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)

def hist(parsed_data, countsperphoton = "Default", thresh = "Default", rng = None, n0 = 0, n1=0):
    """Wrapper for hist_stats, will default to thresholds and counts per photon used when creating parsedData object, but can also manually specify values."""
    if countsperphoton == "Default":
        countsperphoton = parsed_data.countsperphoton
    if thresh == "Default":
        thresh = parsed_data.thresh
    print("Current File: " + str(parsed_data.fnum))
    return hist_stats_roi(parsed_data.roisums, thresh = thresh, countsperphoton = countsperphoton, rng=rng, n0 = n0, n1=n1)

def getLossData(parsed_data, rois = -1, plots = False, timeOrdered = False, survival = False):
    """Returns sorted keyvals, loss between pairs of images in experiment, and error in that measurement, in that order. If data from individual roi is wanted, specify which roi number with indroi input."""

    if timeOrdered:
#         if parsed_data.roisums.shape[0]!=1:
#             raise ValueError('Cannot plot fill per cycle with variations.')
        roisums = parsed_data.roisums.reshape(parsed_data.nreps*parsed_data.keySort.shape[0], parsed_data.npics//2, 2, parsed_data.masks.shape[0])
#         roisums = roisums.swapaxes(0,1)
        ks = range(roisums.shape[0])
    else:
        roisums = parsed_data.roisums
        ks = parsed_data.keySort
    losses =[]
    losserrs = []
    infids = []
    for var in range(roisums.shape[0]):
        infidelity, inf_err, lossfrac, loss_err = hist_stats_roi(roisums, i=var, rois = rois, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, quiet = True, plots = plots)
        losses.append(lossfrac)
        infids.append(infidelity)
        losserrs.append(loss_err)

    if survival:
        return ks, (100-arr(losses))/100, arr(losserrs)/100
    else:
        return ks, arr(losses), arr(losserrs)

def indPhases(parsed_data, plot = True, f = 1, modnum = 1, keepind = -1):
#         centers = []
#         errors = []
#         widths = []
    amps = []
    phases = []
    offsets = []
    for rois, mask in enumerate(parsed_data.masks):
        losses =[]
        losserrs = []
        infids = []
        for var in range(parsed_data.roisums.shape[0]):
            infidelity, inf_err, lossfrac, loss_err = hist_stats_roi(parsed_data.roisums,i=var, thresh = parsed_data.thresh, countsperphoton = parsed_data.countsperphoton, rois = rois, quiet = True, plots = False)
            losses.append(lossfrac)
            infids.append(infidelity)
            losserrs.append(loss_err)
#             maxcoord = ks[np.argmax(losses)]

        if modnum == 1:
            losses = arr(losses)
            ks = arr(parsed_data.keySort)
            losserrs = arr(losserrs)

        else:
#             losses = np.mean(arr([losses[:12], losses[12:]]), axis = 0)
            losses = np.mean(np.split(arr(losses),modnum), axis = 0)
            ks = np.split(arr(parsed_data.keySort),modnum)[0][:,0]
            losserrs = np.mean(np.split(arr(losserrs),modnum), axis = 0)
#             print(np.shape(losses), np.shape(ks),np.shape(losserrs))

        if type(keepind)==type([0]):
            losses = losses[keepind]
            ks = ks[keepind]
            losserrs = losserrs[keepind]

        ks = ks+0.75

#         ks = ks[losses>0]
#         losserrs = losserrs[losses>0]
#         losses = losses[losses>0]
#             try:
#             popt, pcov = gausfit(ks, losses,y_offset=True, negative=False, guess = [79.68, 50, .04, 10])
#            popt, pcov = gausfit(ks, losses,y_offset=True, negative=True)
#             f = 0.9
        popt, pcov = cosFitF(ks, losses, f)
#                 popt, pcov = cosFit(ks, losses)

#             except:
#                 print('fit failed')
# #                 popt = [0, 0, 0, 0]
# #                 pcov = [popt, popt, popt, popt]
#                 popt = [0, 0, 0]
#                 pcov = [popt, popt, popt]
        xs = np.linspace(np.min(ks), np.max(ks), 100)
#             plt.plot(xs, gaussian(xs, *popt), 'r-')
        plt.plot(xs, cos(xs, f, *popt), 'r-')
        perr = np.sqrt(np.diag(pcov))

        amps.append(popt[0])
        phases.append(popt[1])
        offsets.append(popt[2])

#             centers.append(popt[0])
#             errors.append(perr[0])
#             widths.append(popt[2])
#             amps.append(popt[1])
        plt.errorbar(ks, (losses), yerr=losserrs)
        plt.xlabel(parsed_data.keyName)
        plt.ylabel('losses')
        plt.axis([min(xs), max(xs), 0, 100])
        plt.title(rois)

        if plot:
            plt.show()
        else:
            plt.close()
#         return centers, errors, widths, amps
    return amps, phases, offsets



###

def get_threshold(evencounts, oddcounts, tmin = 2, tmax = 10, criteria = 'inf'):
    """Finds threshold that minimizes the infidelity between two images"""
    trial_threshes = np.arange(tmin, tmax, 0.1)
    infs = []
    losses = []
    tots = []
    for thresh in trial_threshes:
        odds = np.array([int(x) for x in oddcounts>thresh])
        evens = np.array([int(x) for x in evencounts>thresh])
        sums = odds + evens
        diffs = evens - odds
        aa = len(sums[sums > 1])
        vv = len(sums[sums < 1])
        av = len(sums[diffs == 1])
        va = len(sums[diffs < 0])
        infs.append(va)
        losses.append(2*(av-va))
    tots = np.array(infs)+np.array(losses)
    #plt.plot(trial_threshes, infs)
    #plt.plot(trial_threshes, losses)
    #plt.plot(trial_threshes, tots)
    if criteria == 'inf':
        return trial_threshes[np.argmin(infs)]
    elif criteria == 'tot':
        return trial_threshes[np.argmin(tots)]

# def hist_stats(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, rois = -1, quiet = False, plots = True, rng = None):
#     lossfracs = []
#     infidelities = []
# # roisums_old: (variation, nreps*npics, rois), roisums: (variation, nreps, npics, rois)
# # The issue below is that flatten mode 'F' (column order) is being used s.th. iterating by 2 jumps images and not ROIs. Easy fix for now but keep in mind for eventually handling
# #     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
#     a, b, c, d = roisums.shape
#     roisums = roisums.reshape(a, b*c, d)
#     flat = np.array(roisums[i,:,:]).flatten('F')/countsperphoton
#     if rois >= 0:
#         flat = np.array(roisums[i,:,rois]).flatten('F')/countsperphoton
#     evencounts = flat[0::2]
#     oddcounts = flat[1::2]


#     oddcounts = oddcounts[np.round(evencounts, 2)!=0.0]
#     evencounts = evencounts[np.round(evencounts, 2)!=0.0]

# #     REMOVED. Previously here to account for lost images, this shouldn't happen any more, so removing this is a convenient check for if things are broken.
# #     oddcounts = oddcounts[evencounts>-100]
# #     evencounts = evencounts[evencounts>-100]
# #     print(flat.shape)

#     if thresh==-999:
#         thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)

#     if plots:
#         ne, bse, ps = plt.hist(evencounts,50, normed=True, range = rng, histtype = 'step', color = '0')
#         plt.plot([thresh, thresh],[0, max(ne)],'k--')
#         #plt.show()

#         n, bs, ps = plt.hist(oddcounts,50, normed=True, range = rng, histtype = 'step', color = '0.5', zorder = 0)
#         plt.plot([thresh, thresh],[0, max(n)],'--', color = '0.5', zorder = 0)
#         #plt.axis([-5, 40, 0, 100])

#         plt.xlabel("Photons Collected")
#         plt.ylabel("Normalized Counts")
#         plt.legend(['Even', 'Odd'])
#         plt.show()

#     odds = np.array([int(x) for x in oddcounts>thresh])
#     evens = np.array([int(x) for x in evencounts>thresh])
#     sums = odds + evens
#     diffs = evens - odds
#     aa = len(sums[sums > 1])
#     vv = len(sums[sums < 1])
#     av = len(sums[diffs == 1])
#     va = len(sums[diffs < 0])
#     if not quiet:
#         print('total pairs: ' + str(len(sums)))
#         print('atom atom: ' + str(aa))
#         print('void void: ' + str(vv))
#         print('atom void: ' + str(av))
#         print('void atom: ' + str(va))
#         print(aa,vv,av)
#         print('total even: ', np.sum(evens))
#         print('total odd: ', np.sum(odds))
#         print('total loss: ', (np.sum(evens)-np.sum(odds))/np.sum(evens))

#     tot = aa+vv+av+va
#     ff = (aa+av)/tot # fill fraction
#     if ff > 0:
#         infidelity = np.round(va*100/len(sums), 2)
#         lossfrac = np.round((av-va)*100/(ff*len(sums)), 2)
#         inf_err = np.round(np.sqrt(va)*100/len(sums), 2)

#         z = 1.96 # corresponds to 95% CI
#         if lossfrac>0:
#             loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/len(sums))
#         else:
#             loss_err = 0
#         #loss_err = np.round(np.sqrt(av-va)*100/(ff*len(sums)), 2)
#     else:
# #         print('fill fraction = 0')
#         return -1, -1, -1, -1
#     lossfracs.append(lossfrac)
#     infidelities.append(infidelity)
#     if not quiet:
#         print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
#         print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
#         print('even/odd thresholds: ',thresh,thresh)
#     return infidelity, inf_err, lossfrac, loss_err

def hist_stats_roi(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, rois = -1, quiet = False, plots = True, rng = None, n0 = 0, n1 = 0, returnFullData = False):
    lossfracs = []
    infidelities = []
# roisums_old: (variation, nreps*npics, rois), roisums: (variation, nreps, npics, rois)
# The issue below is that flatten mode 'F' (column order) is being used s.th. iterating by 2 jumps images and not ROIs. Easy fix for now but keep in mind for eventually handling
#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    nVar, nRep, nPic, nRoi = roisums.shape

    oddcounts = roisums[i,:,1,:]/countsperphoton
    evencounts = roisums[i,:,0,:]/countsperphoton

    if type(rois) == int or type(rois) == np.int32:
        if rois >= 0:
            oddcounts = roisums[i,:,1,rois]/countsperphoton
            evencounts = roisums[i,:,0,rois]/countsperphoton
            nRoi = 1
    else:
            oddcounts = roisums[i,:,1,rois]/countsperphoton
            evencounts = roisums[i,:,0,rois]/countsperphoton
            nRoi = len(rois)

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)

    if plots:
        ne, bse, ps = plt.hist(evencounts.flatten(),50, density=True, range = rng, histtype = 'step', color = '0')
        plt.plot([thresh, thresh],[0, max(ne)],'k--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts.flatten(),50, density=True, range = rng, histtype = 'step', color = '0.5', zorder = 0)
        plt.plot([thresh, thresh],[0, max(n)],'--', color = '0.5', zorder = 0)
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
        plt.legend(['Even', 'Odd'])
        plt.savefig('hist.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()


    odds = np.greater(oddcounts, thresh).astype(int)
    evens = np.greater(evencounts, thresh).astype(int)

    sums = odds + evens
    diffs = evens - odds

    aa = np.greater(sums, 1).astype(int)
    vv = np.less(sums, 1).astype(int)
    av = np.equal(diffs, 1).astype(int)
    va = np.less(diffs, 0).astype(int)

    if not n0 == 0:
        av = av.reshape(nRep, n0, n1)
        va = va.reshape(nRep, n0, n1)

        avxp = np.roll(av, 1, axis = 1)
        avxp[:, 0, :] = 0
        avxm = np.roll(av, -1, axis = 1)
        avxm[:, -1, :] = 0
        avyp = np.roll(av, 1, axis = 2)
        avyp[:, :, 0] = 0
        avym = np.roll(av, -1, axis = 2)
        avym[:, :, -1] = 0

        av_va = (avxp + avxm + avyp + avym)*va

    aaf, vvf, avf, vaf = np.sum(aa), np.sum(vv), np.sum(av), np.sum(va) #sum over ROIs and repetitions.
    npair = len(sums.flatten())
    if not quiet:
        print('total pairs: ' + str(npair))
        print('atom atom: ' + str(aaf))
        print('void void: ' + str(vvf))
        print('atom void: ' + str(avf))
        print('void atom: ' + str(vaf))
        print('total even: ', np.sum(evens))
        print('total odd: ', np.sum(odds))
        print('total loss: ', (np.sum(evens)-np.sum(odds))/np.sum(evens))

    tot = aaf+vvf+avf+vaf
    ff = (aaf+avf)/tot # fill fraction
    if ff > 0:
        infidelity = np.round(vaf*100/npair, 2)
        lossfrac = np.round((avf-vaf)*100/(ff*npair), 2)

#         lossfrac = np.round((avf)*100/(ff*npair), 2) #This is more conservative estimate of loss without considering VA.

        inf_err = np.round(np.sqrt(vaf)*100/npair, 2)

        z = 1.96 # corresponds to 95% CI
        if lossfrac>0:
            loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/npair)
        else:
            loss_err = 0
        #loss_err = np.round(np.sqrt(avf-vaf)*100/(ff*len(sums)), 2)
    else:
#         print('fill fraction = 0')
        return -1, -1, -1, -1
    lossfracs.append(lossfrac)
    infidelities.append(infidelity)
    if not quiet:
        print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
        print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
        print('even/odd thresholds: ',thresh,thresh)

#     plt.imshow(np.sum(av, axis = 0))
#     plt.title('av')
#     plt.colorbar()
#     plt.show()

#     plt.imshow(np.sum(va, axis = 0))
#     plt.title('va')
#     plt.colorbar()
#     plt.show()

#     plt.imshow(np.sum(av_va, axis = 0))
#     plt.title("av_va")
#     plt.colorbar()
#     plt.show()
    if not returnFullData:
        return infidelity, inf_err, lossfrac, loss_err
    else:
        return aa, vv, av, va

def correlationsNN(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, rois = -1, quiet = False, plots = True, rng = None):
    lossfracs = []
    infidelities = []
# roisums_old: (variation, nreps*npics, rois), roisums: (variation, nreps, npics, rois)
# The issue below is that flatten mode 'F' (column order) is being used s.th. iterating by 2 jumps images and not ROIs. Easy fix for now but keep in mind for eventually handling
#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    nVar, nRep, nPic, nRoi = roisums.shape
    oddcounts = roisums[i,:,1,:]/countsperphoton
    evencounts = roisums[i,:,0,:]/countsperphoton

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)

    if plots:
        ne, bse, ps = plt.hist(evencounts.flatten(),50, normed=True, range = rng, histtype = 'step', color = '0')
        plt.plot([thresh, thresh],[0, max(ne)],'k--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts.flatten(),50, normed=True, range = rng, histtype = 'step', color = '0.5', zorder = 0)
        plt.plot([thresh, thresh],[0, max(n)],'--', color = '0.5', zorder = 0)
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
        plt.legend(['Even', 'Odd'])
#         plt.savefig('hist.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()


    odds = np.greater(oddcounts, thresh).astype(int)
    evens = np.greater(evencounts, thresh).astype(int)

    sums = odds + evens
    diffs = evens - odds

    aa = np.greater(sums, 1).astype(int)
    vv = np.less(sums, 1).astype(int)
    av = np.equal(diffs, 1).astype(int)
    va = np.less(diffs, 0).astype(int)

    N = np.sqrt(nRoi).astype(int)
    av = av.reshape(nRep, N, N)
    va = va.reshape(nRep, N, N)

    avxp = np.roll(av, 1, axis = 1)
    avxp[:, 0, :] = 0
    avxm = np.roll(av, -1, axis = 1)
    avxm[:, -1, :] = 0
    avyp = np.roll(av, 1, axis = 2)
    avyp[:, :, 0] = 0
    avym = np.roll(av, -1, axis = 2)
    avym[:, :, -1] = 0

    vaxp = np.roll(va, 1, axis = 1)
    vaxp[:, 0, :] = 0
    vaxm = np.roll(va, -1, axis = 1)
    vaxm[:, -1, :] = 0
    vayp = np.roll(va, 1, axis = 2)
    vayp[:, :, 0] = 0
    vaym = np.roll(va, -1, axis = 2)
    vaym[:, :, -1] = 0

    aaf, vvf, avf, vaf = np.sum(aa), np.sum(vv), np.sum(av), np.sum(va)
    npair = len(sums.flatten())
    if not quiet:
        print('total pairs: ' + str(npair))
        print('atom atom: ' + str(aaf))
        print('void void: ' + str(vvf))
        print('atom void: ' + str(avf))
        print('void atom: ' + str(vaf))
        print('total even: ', np.sum(evens))
        print('total odd: ', np.sum(odds))
        print('total loss: ', (np.sum(evens)-np.sum(odds))/np.sum(evens))

    tot = aaf+vvf+avf+vaf
    ff = (aaf+avf)/tot # fill fraction

    if ff > 0:
        infidelity = np.round(vaf*100/npair, 2)
        lossfrac = np.round((avf-vaf)*100/(ff*npair), 2)
        inf_err = np.round(np.sqrt(vaf)*100/npair, 2)

        z = 1.96 # corresponds to 95% CI
        if lossfrac>0:
            loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/npair)
        else:
            loss_err = 0
        #loss_err = np.round(np.sqrt(avf-vaf)*100/(ff*len(sums)), 2)
    else:
#         print('fill fraction = 0')
        return NULL

    av_va =   (avxp + avxm + avyp + avym)*va/(av.sum(axis = 0) * va.sum(axis = 0))*nRep
    av_av = (avxp + avxm + avyp + avym)*av/(av.sum(axis = 0) * av.sum(axis = 0))*nRep
    va_va = (vaxp + vaxm + vayp + vaym)*va/(va.sum(axis = 0) * va.sum(axis = 0))*nRep
    va_av =   (vaxp + vaxm + vayp + vaym)*av/(va.sum(axis = 0) * av.sum(axis = 0))*nRep

#     print(av_av[np.sum(av_av, axis = (1,2))==8][0])

    plt.plot(np.sum(av_va, axis = (1,2)))
    plt.show()

    plt.plot(np.sum(av_av, axis = (1,2)))
    plt.show()

    plt.plot(np.sum(va_va, axis = (1,2)))
    plt.show()

    plt.plot(np.sum(va_av, axis = (1,2)))
    plt.show()

    print(np.argmax(np.sum(av_av, axis = (1,2))))



    lossfracs.append(lossfrac)
    infidelities.append(infidelity)
    if not quiet:
        print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
        print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
        print('even/odd thresholds: ',thresh,thresh)

#     plt.imshow(np.sum(av, axis = 0))
#     plt.title('av')
#     plt.colorbar()
#     plt.show()

#     plt.imshow(np.sum(va, axis = 0))
#     plt.title('va')
#     plt.colorbar()
#     plt.show()

#     plt.imshow(np.sum(av_va, axis = 0))
#     plt.title("av_va")
#     plt.colorbar()
#     plt.show()

#     return infidelity, inf_err, lossfrac, loss_err
    return av, va, av_va, av_av, va_va, va_av

def img_stats(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, rois = -1, quiet = False, plots = True):
    lossfracs = []
    infidelities = []

#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    flat = np.array(roisums[i,:,:]).flatten('F')/countsperphoton
    if rois >= 0:
        flat = np.array(roisums[i,:,rois]).flatten('F')/countsperphoton
    evencounts = flat[0::2]
    oddcounts = flat[1::2]


#     oddcounts = oddcounts[np.round(evencounts, 2)!=0.0]
#     evencounts = evencounts[np.round(evencounts, 2)!=0.0]

#     REMOVED. Previously here to account for lost images, this shouldn't happen any more, so removing this is a convenient check for if things are broken.
#     oddcounts = oddcounts[evencounts>-100]
#     evencounts = evencounts[evencounts>-100]
#     print(flat.shape)

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)

    if plots:
        ne, bse, ps = plt.hist(evencounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(ne)],'--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(n)],'--')
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
    #     plt.show()

    odds = np.array([oddcounts>thresh])
    evens = np.array([evencounts>thresh])

    fill = evens
    losses = evens&~odds


#     sums = odds + evens
#     diffs = evens - odds
#     aa = len(sums[sums > 1])
#     vv = len(sums[sums < 1])
#     av = len(sums[diffs == 1])
#     va = len(sums[diffs < 0])
#     if not quiet:
#         print('total pairs: ' + str(len(sums)))
#         print('atom atom: ' + str(aa))
#         print('void void: ' + str(vv))
#         print('atom void: ' + str(av))
#         print('void atom: ' + str(va))
#         print(aa,vv,av)

#     tot = aa+vv+av+va
#     ff = (aa+av)/tot # fill fraction
#     if ff > 0:
#         infidelity = np.round(va*100/len(sums), 2)
#         lossfrac = np.round((av-va)*100/(ff*len(sums)), 2)
#         inf_err = np.round(np.sqrt(va)*100/len(sums), 2)
#         loss_err = np.round(np.sqrt(av+va)*100/(ff*len(sums)), 2)
#     else:
#         print('fill fraction = 0')
#         return -1, -1, -1, -1
#     lossfracs.append(lossfrac)
#     infidelities.append(infidelity)
#     if not quiet:
#         print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
#         print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
#         print('even/odd thresholds: ',thresh,thresh)
#     return infidelity, inf_err, lossfrac, loss_err
    return fill, losses

def spect_plot(keysorts, counts, points, guess):
    """Triple lorentz fit with convenient printing. Guess params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
#     guess = [.2, .25, .03, .05, .05, 79.98, .2, 0]

    ks = keysorts[0]
    kss = np.linspace(np.min(ks),np.max(ks),1000)
    cts = np.mean(counts, axis = 0)
    pts = np.sum(points, axis = 0)
    z = 1.96 # corresponds to 95% CI
    error = z*np.sqrt(cts*(1-cts)/pts)

    popt, pcov = curve_fit(triplor, ks, cts, p0 = guess, maxfev=100000)

#     guess = [.1, .1, .1, .1, .1, .05, .05, 80.02, .2, 0]
#     popt, pcov = curve_fit(fivelor, ks, cts, p0 = guess, maxfev=100000000)

    x = (popt[-3]-ks)*1e3
    xs = np.linspace(np.max(x),np.min(x),1000)

#     n=np.max(cts)
    n=1

#     plt.errorbar(x, cts, yerr = error,fmt='.')
#     plt.plot(xs, triplor(kss, *popt))

    # plt.plot([popt[5], popt[5]],[0, .15], 'r-')
    # plt.plot([popt[5]-popt[6], popt[5]-popt[6]],[0, .15], 'r-')
    # plt.plot([popt[5]+popt[6], popt[5]+popt[6]],[0, .15], 'r-')

    print("Blue SB:")
    print(np.abs(popt[0]))
    print('+-')
    print(np.sqrt(pcov[0][0]))

    print("Red SB:")
    print(np.abs(popt[2]))
    print('+-')
    print(np.sqrt(pcov[2][2]))

    print("ratio:")
    print(np.abs(popt[0]/popt[2]))
    print('ratio bound:')
    print(np.abs(popt[0])/(np.abs(popt[2])+np.sqrt(pcov[2][2])))

    print("splitting:")
    print(str(1000*popt[6])+' kHz')
    print('+-')
    print(str(1000*np.sqrt(pcov[6][6]))+' kHz')
    print(popt)
    return x, cts/n, error/n, xs, triplor(kss, *popt)/n

def find_rois(dat, roi_size, filter_size, threshold, bg_offset, display=True):
    """Find peaks in image and define set of ROIs. Returns rois, bgrois."""
    # dat=np.mean(dat,axis=0)
    # dat=ndi.gaussian_filter(dat,1.5)
    dmin=ndi.filters.minimum_filter(dat,filter_size)
    dmax=ndi.filters.maximum_filter(dat,filter_size)

    maxima = (dat==dmax)
    diff = ((dmax-dmin)>threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndi.label(maxima)
    xys = np.array(ndi.center_of_mass(dat, labeled, range(1, num_objects+1)))

#     sort = np.lexsort((xys[:,0],xys[:,1]))
#     xys = xys[sort]

#     keep = [True]
#     for i in range(len(xys)-1):
#         xy = xys[i]
#         xyp = xys[i+1]
#         if np.abs(xyp[0]-xy[0])<roi_size/10:
# #         or np.abs(xyp[1]-xy[1])<roi_size/5:
#             keep.append(False)
#         else:
#             keep.append(True)

#     xys = xys[keep]

    rois=[[int(xy[1])-roi_size, int(xy[1])+roi_size, int(xy[0])-roi_size, int(xy[0])+roi_size] for xy in xys]
    bgrois = [[roi[0]+bg_offset[0],roi[1]+bg_offset[0],roi[2]+bg_offset[1],roi[3]+bg_offset[1]] for roi in rois]

    corners = arr(rois)[:,[0,2]]

    for i in range(len(corners)-1):
        if corners[i+1,1]-corners[i,1] < roi_size:
            corners[i+1,1] = corners[i,1]

# #     Find average separation.
#     xvals = corners[:,0].sort()
#     diff = [y - x for x, y in zip(*[iter(xvals)] * 2)]
#     davg = sum(diff) / len(diff)

    # corners = (corners/roiSettings[0]).astype(int)
#     corners = np.fix((corners-[min(corners[:,0]), min(corners[:,1])])/(roi_size))
    sort = np.lexsort((corners[:,0], corners[:,1]))
#     sort0 = np.argsort(corners[:,1])

    rois = arr(rois)[sort]
    bgrois = arr(bgrois)[sort]
    xys = xys[sort]

    rois = rois[corners[:,0]>10]
    corners = corners[corners[:,0]>10]
    rois = rois[corners[:,1]<dat.shape[0]-10]

    if display:
        fig, ax = plt.subplots(figsize = (15,15))
        ax.imshow(dat)
#         ax = plt.gca()
        for i in range(len(rois)):
            roi = rois[i]
            ax.add_patch(
            patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
            ax.add_patch(
            patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))

            plt.text(xys[i,1],xys[i,0],str(i), color = 'white')
        # plt.plot(xys[:, 1], xys[:, 0], 'r.')
        plt.show()
    print("Peaks found:" + str(len(rois)))
    return rois, bgrois


def get_projections(image, roi):
    """Bin image within roi in x and y directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis = 0)
    image = image[roi[2]:roi[3],roi[0]:roi[1]]
    #image = image - ndi.filters.gaussian_filter(image, sigma=10)
    x_proj = np.sum(image, axis = 0)
    y_proj = np.sum(image, axis = 1)
    #plt.plot(x_proj)
    #plt.plot(y_proj)
    return x_proj, y_proj

# def findroi(dat, roi_size, bg_offset, display=True):
#     """Automatically define ROI around brightest pixel in image, and offset background ROI"""
#     mdat=np.mean(dat, axis=0)
#     i=mdat.argmax()
#     rows=range(mdat.shape[1])
#     cols=range(mdat.shape[0])
#     x,y=np.meshgrid(rows,cols)
#     x,y=x.flatten(),y.flatten()
#     x0=x[i]
#     y0=y[i]

#     roi=[x0-roi_size, x0+roi_size, y0-roi_size, y0+roi_size]

#     if display:
#         plt.imshow(mdat)
#         ax1 = plt.gca()
#         ax1.add_patch(
#         patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
#         ax1.add_patch(
#         patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
#         plt.show()
#     return roi


#
# returns a list of rois, for looking at multiple spots
#
#

def get_rois(n, m, spot_offset, x0, y0, w):
    rois = []
    for i in range(n):
        for ii in range(m):
            xi = int(x0+i*spot_offset +ii*spot_offset)
            yi = int(y0+i*spot_offset -ii*spot_offset)
            roi = [xi, xi + w, yi, yi + w]
            rois.append(roi)
    return rois


#
#  Calculates the difference in roi sums between two regions on an image.  roi [x1, x2, y1, y1] defines the positive
#  value.  Offset (x_o, y_o) defines the offset of the background region from the image region.  If passed a single
#  image, will use that.  If passed a list of images, will calculate the mean image, then proceed with processing the
#  resulting image.
#

def get_roi_sum(image, roi, bg_offset, display=True, bgsub = False):
    """Get sum in rectangular region of image, with the option of displaying the region of interest over a plot of the data."""
    #if len(image.shape) == 3:
    #    image = np.mean(image, axis = 0)
    if display:
        plt.imshow(image)
        ax1 = plt.gca()
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color= 'black'))
        ax1.add_patch(
        patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    #imsum = np.sum(image[roi[0]:roi[1],roi[2]:roi[3]])
    #bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
    imsum = np.sum(image[roi[2]:roi[3],roi[0]:roi[1]])
    if bgsub:
        bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
        return imsum - bgsum
    return imsum


def get_max(image, roi, bg_offset, display=True, bgsub = False):
    """Get sum in rectangular region of image, with the option of displaying the region of interest over a plot of the data."""
    #if len(image.shape) == 3:
    #    image = np.mean(image, axis = 0)
    if display:
        plt.imshow(image)
        ax1 = plt.gca()
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
        ax1.add_patch(
        patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    #imsum = np.sum(image[roi[0]:roi[1],roi[2]:roi[3]])
    #bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
    immax = max(image[roi[2]:roi[3],roi[0]:roi[1]])
    return immax

def var_scan_atom_sumcounts(exp, run, masks, threshold):
    a_cs_ls = []
    pics = exp.pics[::2]
    b = get_binarized(exp, run, masks, threshold=threshold)[::2]
    bkg = np.mean(pics[:, 0:7, 0:7])

    for i in range(len(exp.key)):
        pics_var = pics[i*exp.reps : (i + 1)*exp.reps]
        b_var = b[i*exp.reps : (i + 1)*exp.reps]
        cs = []
        for i in range(len(masks)):
            m = masks[i]
            sig = pics_var[b_var[:,i]>0.5]
            if (len(sig) > 0):
                diff = [np.subtract(s, bkg) for s in sig]
                cs.append(np.mean([np.sum(d) for d in diff*m]))
        a_cs_ls.append(np.mean(cs))
    a_cs_ls = np.array(a_cs_ls)
    a_cs_ls_sorted = np.array(a_cs_ls)[np.argsort(exp.key)]

    key_sorted = np.sort(exp.key)

    fig, ax = plt.subplots(figsize=[5,4])
    plt.plot(key_sorted, a_cs_ls_sorted, 'ko', alpha=0.7)
    plt.xlabel(exp.key_name)
    plt.ylabel('mean atom counts')
    plt.title(exp.data_addr + "data_" + str(run) + ".h5")
    plt.show()


# Need to fix this function to return counts in each variation, right now it's
# summing counts over all images
def var_scan_sumcounts(exp, run, masks, fit='none'):
    cs_ls = []
    pics = exp.pics[::2]

    for m in masks:
        sig = exp.pics[:]
        bkg = np.mean(exp.pics[:, 0:7, 0:7])
        diff = [np.subtract(s, bkg) for s in sig]
        cs = [np.sum(d) for d in diff*m]
        cs_ls.append(cs)

    cs_ls = np.sum(cs_ls, axis=0)
    cs_ls_sorted = np.array(cs_ls)[np.argsort(exp.key)]

    key_sorted = np.sort(exp.key)

    if fit == 'gaussian_peak':
        pguess = [np.max(patom_sorted)-np.min(patom_sorted),
                  key_sorted[np.argmax(patom_sorted)],
                  (key_sorted[-1]-key_sorted[0])/3,
                  np.min(patom_sorted)]
        popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)

        print('key fit = {:.3e}'.format(popt[1]))

    if fit == 'gaussian_dip':
        pguess = [-np.max(patom_sorted)+np.min(patom_sorted),
                  key_sorted[np.argmin(patom_sorted)],
                  (key_sorted[-1]-key_sorted[0])/3,
                  np.max(patom_sorted)]
        popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)

        print('key fit = {:.3e}'.format(popt[1]))


    if fit == 'hockey':
        pguess = [(key_sorted[-1]+key_sorted[0])/2,
                  (np.max(patom_sorted)-np.min(patom_sorted))/ (key_sorted[-1]-key_sorted[0]),
                  np.max(patom_sorted)]
        popt, pcov = curve_fit(hockey, key_sorted, patom_sorted, p0=pguess)

        print('key fit = {:.3e}'.format(popt[0]))

    key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
    fig, ax = plt.subplots(figsize=[5,4])
    if fit == 'gaussian_peak' or fit == 'gaussian_dip':
        plt.plot(key_fine, gaussian(key_fine, *popt), 'k-')
    if fit == 'hockey':
        plt.plot(key_fine, hockey(key_fine, *popt), 'k-')
    plt.plot(key_sorted, cs_ls_sorted, 'ko', alpha=0.7)
    plt.xlabel(exp.key_name)
    plt.ylabel('summed counts')
    plt.title(exp.data_addr + "data_" + str(run) + ".h5")
    plt.show()

def gaussian(x, *p):
    return p[0] * np.exp( -((x - p[1])**2) / (2*(p[2]**2)) ) + p[3]

def double_gaussian(x, *p):
    return p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )

def erfc(x, amp, x0, sigma):
    if x < x0:
        return amp/2 * (1 + erf((x-x0)/np.sqrt(2)/np.abs(sigma)))
    if x >= x0:
        return amp/2 * (1 - erf((x-x0)/np.sqrt(2)/np.abs(sigma)))

def hockey(x_arr, x0, rate, offset):
    y_arr =[]
    for x in x_arr:
        if x < x0:
            y_arr.append(offset)
        else:
            y_arr.append(offset - rate*(x-x0))
    return y_arr

def find_threshold(exp, run, masks, threshold_guess = 10, bin_width = 4, fit = True, surv = True, crop = [0,None,0,None], output = True):

    hist_xdata_all = []
    hist_all = []
    threshold_all = []
    popt_all = []

    numarr = []
    css = []

    if (surv):
        numarr = [0,1]
    else:
        numarr = [0]

    for num in numarr:
        cs = []
        cut = 10
        bkg = (np.mean(exp.pics[:, :cut, :-cut]) + np.mean(exp.pics[:, :-cut, -cut:]) + np.mean(exp.pics[:, -cut:, cut:]) + np.mean(exp.pics[:, cut:, :cut]))/4

        for m in masks:
            sig = exp.pics[num::2, crop[0]:crop[1], crop[2]:crop[3]]
            diff = sig-bkg #[np.subtract(sig[i], bkg) for i in range(len(sig))]
            sum_masked = [np.sum(d) for d in diff*m]
            [cs.append(c) for c in sum_masked]

        css.append(cs)

        hist, bin_edges = np.histogram(cs, bins=np.arange(np.min(cs), np.max(cs), bin_width))
        hist_xdata = np.add(bin_edges, (bin_edges[1]-bin_edges[0])/2)[:-1]

        hist_xdata_all.append(hist_xdata)
        hist_all.append(hist)

        if (fit):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = [np.max(hist[:idx_guess])-np.min(hist[:idx_guess]),
                      hist_xdata[np.argmax(hist[:idx_guess])],
                      (hist_xdata[idx_guess]-hist_xdata[0])/3,
                      np.max(hist[idx_guess:])-np.min(hist[idx_guess:]),
                      hist_xdata[np.argmax(hist[idx_guess:])],
                      (hist_xdata[-1]-hist_xdata[idx_guess])/3]
            popt, pcov = curve_fit(double_gaussian, hist_xdata, hist, p0=pguess)
            popt_all.append(popt)

            hist_xdata_fine = np.linspace(hist_xdata[0], hist_xdata[-1], 200, endpoint=True)

            mistake_arr = []

            t_arr = np.linspace(popt[1], popt[4], 100)
            for t in t_arr:

                atom_mistake = erfc(t, popt[3], popt[4], popt[5])
                void_mistake = erfc(t, popt[0], popt[1], popt[2])
                mistake = void_mistake + atom_mistake
                mistake_arr.append(mistake)

            threshold = t_arr[np.argmin(mistake_arr)]
            min_mistake = np.min(mistake_arr)

            if num == 0:
                infidelity = min_mistake/(popt[0]*popt[2]*np.sqrt(2*np.pi)+popt[3]*popt[5]*np.sqrt(2*np.pi))

        else:
            threshold = threshold_guess
            min_mistake = 1000

        threshold_all.append(threshold)

    if output == True:
        fig, ax = plt.subplots(figsize=[6,4])
        plt.plot(hist_xdata_all[0], hist_all[0], alpha=0.9, drawstyle='steps', color='k', zorder = 0)
        plt.plot([threshold_all[0], threshold_all[0]], [np.min(hist_all[0]), np.max(hist_all[0])], 'k--', alpha=0.9, label='even', zorder = 0)
        if (surv):
            plt.plot(hist_xdata_all[1], hist_all[1], alpha=0.5, drawstyle='steps', color='k', zorder = 0)
            plt.plot([threshold_all[1], threshold_all[1]], [np.min(hist_all[1]), np.max(hist_all[1])], 'k--', alpha=0.5, label='odd', zorder = 0)
        if (fit and not surv):
           plt.plot(hist_xdata_fine, double_gaussian(hist_xdata_fine, *popt), 'b-', alpha=0.4, zorder = 0)
        plt.xlabel('Counts collected')
        plt.ylabel('Events')
        plt.ylim(0, )
        plt.legend()
        plt.title(exp.data_addr + "data_" + str(run) + ".h5")
        plt.show()


        print('even/odd atom peak position: '+'{:.3f}'.format(popt_all[0][4])+'/{:.3f}'.format(popt_all[1][4]))
        print('even/odd bkg peak width: '+'{:.3f}'.format(abs(popt_all[0][2]))+'/{:.3f}'.format(abs(popt_all[1][2])))
        print('even/odd atom peak width: '+'{:.3f}'.format(abs(popt_all[0][5]))+'/{:.3f}'.format(abs(popt_all[1][5])))
        print('even/odd thresholds: '+'{:.3f}'.format(threshold_all[0])+'/{:.3f}'.format(threshold_all[1]))
        print('fitted infidelity: '+'{:.3f}'.format(infidelity*100)+' percent')

    return threshold_all[0], threshold_all[1], popt_all, hist_xdata_all, hist_all, min_mistake, css, infidelity

# def find_threshold_2(exp, run, roi=[12, 15, 11, 14], threshold_guess = 30, bin_number = 100):
#     sig = exp.pics[:, roi[0]:roi[1], roi[2]:roi[3]]
#     bkg = np.mean(exp.pics[:, 0:7, 0:7])
#     diff = [np.subtract(s, bkg) for s in sig]
#     cs = [np.sum(d) for d in diff]
    ############################################
#     hist_1, bin_edges_1 = np.histogram(cs[::2], bins=bin_number)
#     hist_xdata_1 = np.add(bin_edges_1, (bin_edges_1[1]-bin_edges_1[0])/2)[:-1]

#     idx_guess = np.abs(hist_xdata_1 - threshold_guess).argmin()

#     pguess_1 = [np.max(hist_1[:idx_guess])-np.min(hist_1[:idx_guess]),
#               hist_xdata_1[np.argmax(hist_1[:idx_guess])],
#               (hist_xdata_1[idx_guess]-hist_xdata_1[0])/3,
#               np.max(hist_1[idx_guess:])-np.min(hist_1[idx_guess:]),
#               hist_xdata_1[np.argmax(hist_1[idx_guess:])],
#               (hist_xdata_1[-1]-hist_xdata_1[idx_guess])/3,
#               np.min(hist_1)]
#     popt_1, pcov_1 = curve_fit(double_gaussian, hist_xdata_1, hist_1, p0=pguess_1)
#
#     hist_xdata_fine_1 = np.linspace(hist_xdata_1[0], hist_xdata_1[-1], 200, endpoint=True)

#     mistake_arr_1 = []
#     t_arr_1 = np.linspace(popt_1[1], popt_1[4], 100)
#     for t in t_arr_1:

#        atom_mistake = erfc(t, popt_1[3], popt_1[4], popt_1[5])
#        void_mistake = erfc(t, popt_1[0], popt_1[1], popt_1[2])
#        mistake = void_mistake + atom_mistake
#        mistake_arr_1.append(mistake)

#    threshold_1 = t_arr_1[np.argmin(mistake_arr_1)]
    ############################################
#    hist_2, bin_edges_2 = np.histogram(cs[1::2], bins=bin_number)
#    hist_xdata_2 = np.add(bin_edges_2, (bin_edges_2[1]-bin_edges_2[0])/2)[:-1]

#    idx_guess = np.abs(hist_xdata_1 - threshold_guess).argmin()

#    pguess_2 = [np.max(hist_2[:idx_guess])-np.min(hist_2[:idx_guess]),
#              hist_xdata_2[np.argmax(hist_2[:idx_guess])],
#              (hist_xdata_2[idx_guess]-hist_xdata_1[0])/3,
#              np.max(hist_2[idx_guess:])-np.min(hist_2[idx_guess:]),
#              hist_xdata_2[np.argmax(hist_2[idx_guess:])],
#              (hist_xdata_2[-1]-hist_xdata_2[idx_guess])/3,
#              np.min(hist_2)]
#    popt_2, pcov_2 = curve_fit(double_gaussian, hist_xdata_2, hist_2, p0=pguess_2)

#    hist_xdata_fine_2 = np.linspace(hist_xdata_2[0], hist_xdata_2[-1], 200, endpoint=True)

#    mistake_arr_2 = []
#    t_arr_2 = np.linspace(popt_2[1], popt_2[4], 100)
#    for t in t_arr_2:

#        atom_mistake = erfc(t, popt_2[3], popt_2[4], popt_2[5])
#        void_mistake = erfc(t, popt_2[0], popt_2[1], popt_2[2])
#        mistake = void_mistake + atom_mistake
#        mistake_arr_2.append(mistake)

#    threshold_2 = t_arr_2[np.argmin(mistake_arr_2)]
    ############################################
#    fig, ax = plt.subplots(figsize=[5,4])
#    plt.plot(hist_xdata_fine_1, double_gaussian(hist_xdata_fine_1, *popt_1), 'k-')
#    plt.plot(hist_xdata_1, hist_1, 'ko', alpha=0.7, label='1st img')
#    plt.plot([threshold_1, threshold_1], [np.min(hist_1), np.max(hist_1)], 'r--', alpha=0.9, label='1st img threshold: %.2f' %threshold_1)
#    plt.plot(hist_xdata_fine_2, double_gaussian(hist_xdata_fine_2, *popt_2), 'k:')
#    plt.plot(hist_xdata_2, hist_2, marker='o', c='k', mfc='white', alpha=0.7, linestyle='none', label='2nd img')
#    #plt.plot([threshold_2, threshold_2], [np.min(hist_2), np.max(hist_2)], 'r--', alpha=0.5, label='2nd img threshold: %.2f' %threshold_2)
#    plt.xlabel('counts')
#    plt.legend(bbox_to_anchor=[1, 1])
#    plt.title(exp.data_addr + "data_" + str(run) + ".h5")
#    plt.show()
#
#    return threshold_1 #, popt, void_mistakes, atom_mistakes,

def get_binarized(exp, run, masks, threshold=30, crop=[0,None,0,None]):

    cs_arr = []
    cut = 10
    bkg = (np.mean(exp.pics[:, :cut, :-cut]) + np.mean(exp.pics[:, :-cut, -cut:]) + np.mean(exp.pics[:, -cut:, cut:]) + np.mean(exp.pics[:, cut:, :cut]))/4

    for m in masks:
        sig = exp.pics[:, crop[0]:crop[1], crop[2]:crop[3]]
        diff = [np.subtract(sig[i], bkg) for i in range(len(sig))]
        cs = [np.sum(d) for d in diff*m]
        cs_arr.append(cs)

    binarized_images = []
    for cs in cs_arr:
        binarized_atoms = []
        for c in cs:
            binarized_atoms.append(c > threshold)
        binarized_images.append(np.array(binarized_atoms, dtype=int))

    return np.transpose(binarized_images)

def get_masks(imgc, x0=21, y0=14, dx=7, dy=11, N=[4,4], r=2):
    x = np.arange(len(imgc[0]))
    y = np.arange(len(imgc[:,0]))
    xx, yy = np.meshgrid(x, y)
    masks = []

    for i in range(N[0]):
        for j in range(N[1]):
            mask = np.zeros_like(imgc)
            xm, ym = x0 + i*dx, y0 + j*dy
            mask = np.sqrt((xx-xm)**2 + (yy-ym)**2) < r
            masks.append(mask)

    imgcmask = (imgc - np.mean(imgc[:10, :10]))*np.sum(masks,axis=0)

    plt.imshow(imgcmask)
    return masks

def var_scan_loadprob(exp, run, masks, t=30, fit='none', sortkey=0, crop=[0,None,0,None]):

    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    data = data[::2]

    if (exp.key.ndim > 1):
        key = exp.key[:, sortkey]
    else:
        key = exp.key

    patom = []
    patom_err = []
    for i in range(len(exp.key)):
        p = np.sum(data[i*exp.reps : (i + 1)*exp.reps])/exp.reps/len(masks)
        patom.append(p)
        patom_err.append(np.sqrt(p/exp.reps/len(masks)))


    if (exp.key.ndim > 1):

        patom_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(exp.key), patom))), key=lambda x: [x[0], x[1]]))
        patom_sorted_reshape = np.reshape(patom_sorted[:,2], (len(np.unique(exp.key[:, 0])), len(np.unique(exp.key[:, 1]))))
        patom_uncertainty_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(exp.key),patom_err))), key=lambda x: [x[0], x[1]]))
        patom_sorted_reshape_err = np.reshape(patom_uncertainty_sorted[:,2], (len(np.unique(exp.key[:, 0])), len(np.unique(exp.key[:, 1]))))

        k0min = np.min(patom_sorted[:,0])
        k0max = np.max(patom_sorted[:,0])
        k1min = np.min(patom_sorted[:,1])
        k1max = np.max(patom_sorted[:,1])

        fig, ax = plt.subplots(figsize=[5,4])
        im = plt.imshow(patom_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                   aspect=(k1max - k1min)/(k0max - k0min), vmin=0, vmax=1, origin="lower")
        plt.xlabel(exp.key_name[1])
        plt.ylabel(exp.key_name[0])
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('load_prob')
        plt.title(exp.data_addr + "data_" + str(run) + ".h5")
        plt.show()



    else:
        patom_sorted = np.array(patom)[np.argsort(key)]
        patom_err_sorted = np.array(patom_err)[np.argsort(key)]

        key_sorted = np.sort(key)

        if fit == 'gaussian_peak':
            pguess = [np.max(patom_sorted)-np.min(patom_sorted),
                      key_sorted[np.argmax(patom_sorted)],
                      (key_sorted[-1]-key_sorted[0])/3,
                      np.min(patom_sorted)]
            popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)

            print('key fit = {:.3e}'.format(popt[1]))

        if fit == 'gaussian_dip':
            pguess = [-np.max(patom_sorted)+np.min(patom_sorted),
                      key_sorted[np.argmin(patom_sorted)],
                      (key_sorted[-1]-key_sorted[0])/3,
                      np.max(patom_sorted)]
            popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)

            print('key fit = {:.3e}'.format(popt[1]))


        if fit == 'hockey':
            pguess = [(key_sorted[-1]+key_sorted[0])/2,
                      (np.max(patom_sorted)-np.min(patom_sorted))/ (key_sorted[-1]-key_sorted[0]),
                      np.max(patom_sorted)]
            popt, pcov = curve_fit(hockey, key_sorted, patom_sorted, p0=pguess)

            print('key fit = {:.3e}'.format(popt[0]))

        patom_uncertainty = np.sqrt(exp.reps*len(masks)*patom_sorted)/(exp.reps*len(masks))

        key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
        fig, ax = plt.subplots(figsize=[5,4])
        if fit == 'gaussian_peak' or fit == 'gaussian_dip':
            plt.plot(key_fine, gaussian(key_fine, *popt), 'k-')
        if fit == 'hockey':
            plt.plot(key_fine, hockey(key_fine, *popt), 'k-')
        plt.errorbar(key_sorted, patom_sorted, patom_uncertainty, color='k', marker='o', linestyle=':', alpha=0.7)
        if (type(exp.key_name) == str):
            plt.xlabel(exp.key_name)
        else:
            plt.xlabel(exp.key_name[sortkey])
        plt.ylabel('load prob')
        plt.title(exp.data_addr + "data_" + str(run) + ".h5")
        plt.ylim(0, 1)
        plt.show()

    if (exp.key.ndim > 1):
        return patom_sorted
    else:
        return key_sorted, patom_sorted

def get_loss(exp, run, masks, t, sortkey=0, crop=[0,None,0,None]):

    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    loaddata = data[::2]
    survdata = data[1::2]

    if (exp.key.ndim > 1):
        key = exp.key[:, sortkey]
    else:
        key = exp.key

    aa = 0
    av = 0
    va = 0
    vv = 0
    a = 0

    for i in range(len(key)):

        for j in range(exp.reps):
            for m in range(len(masks)):
                atom1 = loaddata[i*exp.reps +j][m]
                atom2 = survdata[i*exp.reps +j][m]
                if (atom1 and atom2):
                    aa += 1
                elif (atom1 and atom2==0):
                    av += 1
                elif (atom1==0 and atom2):
                    va += 1
                elif (atom1==0 and atom2==0):
                    vv += 1
                if (atom1):
                    a += 1

    npair = aa+av+va+vv

    load = a/npair
    surv = aa/a

    load_err = np.sqrt(a)/npair
    surv_err = surv*(1-surv)

    print(' ')
    print('total pairs: ' + str(npair))
    print('atom atom: ' + str(aa))
    print('void void: ' + str(vv))
    print('atom void: ' + str(av))
    print('void atom: ' + str(va))
    print('load: ' +  '{:.3f}'.format(load*100)+ ' +- '+ '{:.3f}'.format(load_err*100)+ ' percent')
    print('survival: ' +  '{:.3f}'.format(surv*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')
    print('loss: ' +  '{:.3f}'.format((1-surv)*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')
    #print('average infidelity: ' + '{:.3f}'.format(inf*100)+ ' +- '+ '{:.3f}'.format(inf_err*100)+ ' percent')

    return va/npair


def var_scan_survprob(exp, run, masks, t=30, fit='none', sortkey=0, crop=[0,None,0,None]):

    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    loaddata = data[::2]
    survdata = data[1::2]

    surv_prob = []
    surv_prob_uncertainty = []
    popt, pcov = [], []

    key = exp.key

    for i in range(len(key)):
        aa = 0
        a = 0
        for j in range(exp.reps):
            for m in range(len(masks)):
                atom1 = loaddata[i*exp.reps +j][m]
                atom2 = survdata[i*exp.reps +j][m]
                if (atom1 and atom2):
                    aa += 1
                if (atom1):
                    a += 1
        if a:
            p = aa/a
            surv_prob.append(p)
            surv_prob_uncertainty.append(p*(1-p))
        else:
            surv_prob.append(0)
            surv_prob_uncertainty.append(0)


    if (exp.key.ndim > 1):

        surv_prob_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(exp.key), surv_prob))), key=lambda x: [x[0], x[1]]))
        surv_prob_sorted_reshape = np.reshape(surv_prob_sorted[:,2], (len(np.unique(exp.key[:, 0])), len(np.unique(exp.key[:, 1]))))
        surv_prob_uncertainty_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(exp.key),surv_prob_uncertainty))), key=lambda x: [x[0], x[1]]))
        surv_prob_sorted_reshape_err = np.reshape(surv_prob_uncertainty_sorted[:,2], (len(np.unique(exp.key[:, 0])), len(np.unique(exp.key[:, 1]))))

        k0min = np.min(surv_prob_sorted[:,0])
        k0max = np.max(surv_prob_sorted[:,0])
        k1min = np.min(surv_prob_sorted[:,1])
        k1max = np.max(surv_prob_sorted[:,1])

        fig, ax = plt.subplots(figsize=[5,4])
        im = plt.imshow(surv_prob_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                   aspect=(k1max - k1min)/(k0max - k0min), vmin=0, vmax=1, origin="lower")
        plt.xlabel(exp.key_name[1])
        plt.ylabel(exp.key_name[0])
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('surv_prob')
        plt.title(exp.data_addr + "data_" + str(run) + ".h5")
        plt.show()


    else:

        surv_prob_sorted = np.array(surv_prob)[np.argsort(key)]
        surv_prob_uncertainty_sorted = np.array(surv_prob_uncertainty)[np.argsort(key)]

        key_sorted = np.sort(key)

        if (exp.key.ndim == 1):
            if fit == 'gaussian_peak':
                pguess = [np.max(surv_prob_sorted)-np.min(surv_prob_sorted),
                          key_sorted[np.argmax(surv_prob_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, surv_prob_sorted, p0=pguess)

                print('key fit = {:.3e}'.format(popt[1]))

            if fit == 'gaussian_dip':
                pguess = [-np.max(surv_prob_sorted)+np.min(surv_prob_sorted),
                          key_sorted[np.argmin(surv_prob_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, surv_prob_sorted, p0=pguess)

                print('key fit = {:.3e}'.format(popt[1]))

            if fit == 'hockey':
                pguess = [(key_sorted[-1]+key_sorted[0])/2,
                          (np.max(surv_prob_sorted)-np.min(surv_prob_sorted))/ (key_sorted[-1]-key_sorted[0]),
                          np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(hockey, key_sorted, surv_prob_sorted, p0=pguess)

                print('key fit = {:.3e}'.format(popt[0]))

            key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
            fig, ax = plt.subplots(figsize=[5,4])
            if fit == 'gaussian_peak' or fit == 'gaussian_dip':
                plt.plot(key_fine, gaussian(key_fine, *popt), 'k-')
            if fit == 'hockey':
                plt.plot(key_fine, hockey(key_fine, *popt), 'k-')
            plt.errorbar(key_sorted, surv_prob_sorted, surv_prob_uncertainty_sorted, color='k', marker='o', linestyle=':', alpha=0.7)
            if (type(exp.key_name) == str):
                plt.xlabel(exp.key_name)
            else:
                plt.xlabel(exp.key_name[sortkey])
            plt.ylabel('surv prob')
            plt.title(exp.data_addr + "data_" + str(run) + ".h5")
            plt.ylim(0, 1)
            plt.show()

    if (exp.key.ndim > 1):
        return surv_prob_sorted_reshape, surv_prob_sorted_reshape_err, surv_prob
    else:
        return key_sorted, surv_prob_sorted, surv_prob_uncertainty_sorted, popt, pcov

# picture_num = 0,1 for getting counts in first or second image
def getCountsPerAtom(exp, run, masks, picture_num, threshold, sortkey=0, crop=[0,None,0,None]):
    img_bin = get_binarized(exp, run, masks, threshold=threshold, crop=crop)
    img_bin_1 = img_bin[::2]

    cs_arr_2 = []

    for m in masks:
        sig = exp.pics[picture_num::2]
        bkg = np.mean(exp.pics[picture_num::2, 0:7, 0:7], axis=(1,2))
        diff = [np.subtract(sig[i], bkg[i]) for i in range(len(sig))]
        cs = [np.sum(d) for d in diff*m]
        cs_arr_2.append(cs)

    cs_arr_cond = []
    for i, img in enumerate(img_bin_1):
        cs = 0
        num_atoms = np.sum(img) # sum of img is number of atoms in the picture
        for j, m in enumerate(img):
            if m:
                cs = cs + cs_arr_2[j][i]

        # mean counts is total cs divided by number atoms
        if (num_atoms > 0):
            csmean = cs/num_atoms
        else:
            csmean = 0

        # changing cs_arr_cond to be an array of mean atoms counts in each image
        cs_arr_cond.append(csmean)

    cs_arr_cond_sum = [np.sum(cs_arr_cond[i*exp.reps:(i+1)*exp.reps]) for i in range(len(exp.key))] # sum over reps for each variation
    cs_arr_cond_mean = np.array(cs_arr_cond_sum)/(exp.reps) # average over reps for each variation
    cs_arr_cond_sum_sorted = np.array(cs_arr_cond_sum)[np.argsort(exp.key)]
    cs_arr_cond_mean_sorted = np.array(cs_arr_cond_mean)[np.argsort(exp.key)]
    key_sorted = np.sort(exp.key)
    #plt.plot(key_sorted, cs_arr_cond_sum_sorted)
    plt.plot(key_sorted, cs_arr_cond_mean_sorted)
    if (type(exp.key_name) == str):
        plt.xlabel(exp.key_name)
    else:
        plt.xlabel(exp.key_name[sortkey])
    plt.ylabel('mean single atom counts')
    plt.title(exp.data_addr + "data_" + str(run) + ".h5")

    return key_sorted, cs_arr_cond_mean_sorted


def newDay():
    """Creates a new notebook (and folder if missing) for current day, from the file that this funciton is called from."""
    src_dir = os.getcwd()
    src_filename = src_dir.replace('A:\\Yb_data\\','')+'.ipynb'

    date_str = time.strftime('%y%m%d')
#     date_str = '201112'

    try:
        os.mkdir('../Yb_' + date_str)
    except:
        print('Directory already exists, using preexisting directory.')

    dest_dir = '../Yb_' + date_str

    src_file = os.path.join(src_dir, src_filename)
    shutil.copy(src_file,dest_dir) #copy the file to destination dir

    dst_file = os.path.join(dest_dir, src_filename)
    new_dst_file_name = os.path.join(dest_dir, 'Yb_'+date_str+'.ipynb')

    os.rename(dst_file, new_dst_file_name)#rename
    # print(date_str, dest_dir, src_file, dst_file, new_dst_file_name)
#     print(2)

def cameraCountsToPhotons(counts, l=556, useOffset=True):
    qe = exc.QE556
    if (l==399):
        qe = exc.QE399
    if (useOffset == True):
        offset = exc.offset
    else:
        offset = 0

    return (counts - offset)*exc.CMOSsensitivity12bit/qe
