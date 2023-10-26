from .imports import *

from .expfile import *
# from .analysis import *
from .mathutil import *
# from .plotutil import *
from .imagutil import *
import time

import klib.experiment_constants as exc

# from .mako import *
# from .adam import *

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
    plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
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

        print('key fit = {:.4e}'.format(popt[1]))


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
    plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
    plt.show()

def find_threshold_2(exp, run, masks, threshold_guess = 10, bin_width = 4, mode = 'emccd', keep_img = [0,1], crop = [0,None,0,None], output=True, single_threshold=True, center_zero=True):

    hist_xdata_all = []
    hist_all = []
    threshold_all = []
    popt_all = []
    inf_all = []
    loss_all = []

    css = []

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    # analyze each image seperately
    for num in range(num_img):
        cs = []
        cut = 10

        #find backrgound
        bkg = (np.mean(exp.pics[num::num_img, :cut, :-cut]) + np.mean(exp.pics[num::num_img, :-cut, -cut:]) + np.mean(exp.pics[num::num_img, -cut:, cut:]) + np.mean(exp.pics[num::num_img, cut:, :cut]))/4

        #find counts from an image
        sig = exp.pics[num::num_img, crop[0]:crop[1], crop[2]:crop[3]]
        if center_zero and mode=='emccd':
            diff=sig
        elif center_zero and mode=='cmos':
            diff = sig-bkg
        else:
            diff=sig
        cs = np.array(list(map(lambda image: list(map(lambda mask:np.sum(mask*image),masks)),diff)))
        cs = cs.flatten()

        css.append(cs)

        # sort counts into histogram
        hist, bin_edges = np.histogram(cs, bins=np.arange(np.min(cs)-bin_width/2, np.max(cs)+bin_width/2, bin_width))
        hist_xdata = bin_edges[:-1]

        hist_xdata_all.append(hist_xdata)
        hist_all.append(hist)

        #fit double gaussian to histogram
        fit_worked = True
        if (mode=='cmos'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = [np.max(hist[:idx_guess])-np.min(hist[:idx_guess]),
                      hist_xdata[np.argmax(hist[:idx_guess])],
                      (hist_xdata[idx_guess]-hist_xdata[0])/5,
                      np.max(hist[idx_guess:])-np.min(hist[idx_guess:]),
                      hist_xdata[idx_guess+np.argmax(hist[idx_guess:])],
                      (hist_xdata[-1]-hist_xdata[idx_guess])/5]
            try:
                popt, pcov = curve_fit(double_gaussian, hist_xdata, hist, p0=pguess) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
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

                infidelity = min_mistake/(popt[0]*popt[2]*np.sqrt(2*np.pi)+popt[3]*popt[5]*np.sqrt(2*np.pi))
                inf_all.append(infidelity)
                infidelity_loss = erfc(threshold, popt[3], popt[4], popt[5])/(popt[3]*popt[5]*np.sqrt(2*np.pi))
                loss_all.append(infidelity_loss)

            except:
                print('threshold fit failed, using threshold guess')
                fit_worked = False
                threshold = threshold_guess
                min_mistake = 1000

        elif (mode=='emccd'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = np.array([ 1.7, 4.9e3,  0.017,
                                np.max(hist[:idx_guess]),  np.max(hist[idx_guess:]), hist_xdata[idx_guess+np.argmax(hist[idx_guess:])],
                                (hist_xdata[-1]-hist_xdata[idx_guess])/5])
            try:
                popt, pcov = curve_fit(emccd_hist, hist_xdata, hist, p0=pguess, bounds=([1, -np.inf, 1e-10, 0, 0, -np.inf, 0], [100, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
                popt_all.append(popt)

                hist_xdata_fine = np.linspace(hist_xdata[0], hist_xdata[-1], 200, endpoint=True)

                mistake_arr = []

                t_arr = np.linspace(hist_xdata[np.argmax(hist[:idx_guess])], hist_xdata[idx_guess+np.argmax(hist[idx_guess:])], 100)
                for t in t_arr:

                    atom_mistake = integrate.quad(gaussian, -np.inf, t, args=(popt[4], popt[5], popt[6], 0,))[0]
                    void_mistake = integrate.quad(emccd_bkg, t, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]
                    mistake = void_mistake + atom_mistake
                    mistake_arr.append(mistake)


                threshold = t_arr[np.argmin(mistake_arr)]
                min_mistake = np.min(mistake_arr)

                tot_area = integrate.quad(gaussian, -np.inf, np.inf, args=(popt[4], popt[5], popt[6], 0,))[0] + integrate.quad(emccd_bkg, -np.inf, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]

                infidelity = min_mistake/tot_area
                inf_all.append(infidelity)

            except:
                print('threshold fit failed, using threshold guess')
                fit_worked = False
                threshold = threshold_guess
                min_mistake = 1000

        else:
            threshold = threshold_guess
            min_mistake = 1000

        if num == 0:
            if single_threshold==True:
                threshold_0 = threshold
            threshold_all.append(threshold)

        if num != 0:
            if single_threshold==True:
                threshold = threshold_0
            threshold_all.append(threshold)


    if output == True:
        fig, ax = plt.subplots(figsize=[6,4])
        alphaarr = np.linspace(1.0,0.5, num_img)
        for num in keep_img:
            ax.plot(hist_xdata_all[num], hist_all[num], alpha=alphaarr[num], drawstyle='steps-post', color='k', zorder = 0)
            ax.plot([threshold_all[num], threshold_all[num]], [np.min(hist_all[num]), np.max(hist_all[num])], color='k', linestyle='--', alpha=alphaarr[num], label=num, zorder = 0)

        ax.set_xlabel('Counts collected')
        ax.set_ylabel('Events')
        ax.set_ylim(0, )
        plt.legend()
        ax.set_title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.show()

        format_string = "{:.3f}"

        if (mode=='cmos' and fit_worked):
            print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
            print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
            print('bkg peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,2]])
            print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
            print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,0]])
            print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            print('fitted infidelity: ',[format_string.format(number*100) for number in inf_all], 'percent')
            print('loss from infidelity: ',[format_string.format(number*100) for number in loss_all], 'percent')

        elif (mode=='emccd' and fit_worked):
            print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
            print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
            print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,6]])
            print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
            print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            #print('fitted infidelity: ',[format_string.format(number*100) for number in inf_all], 'percent')


    if (mode=='cmos' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all, loss_all
    elif (mode=='emccd' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all
    else:
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css



def find_threshold(exp, run, masks, threshold_guess = 10, bin_width = 4, mode = 'emccd', keep_img = [0,1], crop = [0,None,0,None], output=True, single_threshold=True, center_zero=True, cut=10, xlim=7500, logscale=False):


    hist_xdata_all = []
    hist_all = []
    threshold_all = []
    popt_all = []
    inf_all = []
    loss_all = []

    css = []

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    # analyze each image seperately
    for num in range(num_img):
        cs = []
        cut = cut

        #find backrgound
        bkg = (np.mean(exp.pics[num::num_img, :cut, :-cut]) + np.mean(exp.pics[num::num_img, :-cut, -cut:]) + np.mean(exp.pics[num::num_img, -cut:, cut:]) + np.mean(exp.pics[num::num_img, cut:, :cut]))/4

        #find counts from an image
        sig = exp.pics[num::num_img, crop[0]:crop[1], crop[2]:crop[3]]
        if center_zero and mode=='cmos':
            diff = sig
        else:
            diff=sig
        cs = np.array(list(map(lambda image: list(map(lambda mask:np.sum(mask*image),masks)),diff)))
        cs = cs.flatten()

        css.append(cs)

        # sort counts into histogram
        hist, bin_edges = np.histogram(cs, bins=np.arange(np.min(cs)-bin_width/2, np.max(cs)+bin_width/2, bin_width))
        hist_xdata = bin_edges[:-1]

        hist_xdata_all.append(hist_xdata)
        hist_all.append(hist)

        #fit double gaussian to histogram
        fit_worked = True
        if (mode=='cmos'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = [np.max(hist[:idx_guess])-np.min(hist[:idx_guess]),
                      hist_xdata[np.argmax(hist[:idx_guess])],
                      (hist_xdata[idx_guess]-hist_xdata[0])/5,
                      np.max(hist[idx_guess:])-np.min(hist[idx_guess:]),
                      hist_xdata[idx_guess+np.argmax(hist[idx_guess:])],
                      (hist_xdata[-1]-hist_xdata[idx_guess])/5]
            try:
                popt, pcov = curve_fit(double_gaussian, hist_xdata, hist, p0=pguess) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
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

                infidelity = min_mistake/(popt[0]*popt[2]*np.sqrt(2*np.pi)+popt[3]*popt[5]*np.sqrt(2*np.pi))
                inf_all.append(infidelity)
                infidelity_loss = erfc(threshold, popt[3], popt[4], popt[5])/(popt[3]*popt[5]*np.sqrt(2*np.pi))
                loss_all.append(infidelity_loss)

            except:
                print('threshold fit failed, using threshold guess')
                fit_worked = False
                threshold = threshold_guess
                min_mistake = 1000

        elif (mode=='emccd'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = np.array([ 1.7, 0,  0.017,
                                np.max(hist[:idx_guess]),  np.max(hist[idx_guess:]), hist_xdata[idx_guess+np.argmax(hist[idx_guess:])]-hist_xdata[0],
                                (hist_xdata[-1]-hist_xdata[idx_guess])/5])
            try:
                popt, pcov = curve_fit(emccd_hist, np.subtract(hist_xdata, hist_xdata[0]), hist, p0=pguess, bounds=([1, -np.inf, 1e-10, 0, 0, -np.inf, 0], [100, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
                popt_all.append(popt)

                hist_xdata_fine = np.linspace(hist_xdata[0], hist_xdata[-1], 200, endpoint=True)

                mistake_arr = []

                t_arr = np.subtract(np.linspace(hist_xdata[np.argmax(hist[:idx_guess])], hist_xdata[idx_guess+np.argmax(hist[idx_guess:])], 100), hist_xdata[0])
                for t in t_arr:

                    atom_mistake = integrate.quad(gaussian, -np.inf, t, args=(popt[4], popt[5], popt[6], 0,))[0]
                    void_mistake = integrate.quad(emccd_bkg, t, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]
                    mistake = void_mistake + atom_mistake
                    mistake_arr.append(mistake)


                threshold = t_arr[np.argmin(mistake_arr)]+hist_xdata[0]
                min_mistake = np.min(mistake_arr)

                tot_area = integrate.quad(gaussian, -np.inf, np.inf, args=(popt[4], popt[5], popt[6], 0,))[0] + integrate.quad(emccd_bkg, -np.inf, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]

                infidelity = min_mistake/tot_area
                inf_all.append(infidelity)

            except:
                print('threshold fit failed, using threshold guess')
                fit_worked = False
                threshold = threshold_guess
                min_mistake = 1000

        elif (mode=='emccd_skew'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = np.array([ 1.7, 0,  0.017,
                                np.max(hist[:idx_guess]),  np.max(hist[idx_guess:]), hist_xdata[idx_guess+np.argmax(hist[idx_guess:])]-hist_xdata[0],
                                (hist_xdata[-1]-hist_xdata[idx_guess])/5, 0])
            try:
                popt, pcov = curve_fit(emccd_hist_skew, np.subtract(hist_xdata, hist_xdata[0]), hist, p0=pguess, bounds=([1, -np.inf, 1e-10, 0, 0, -np.inf, 0, -np.inf], [100, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
                popt_all.append(popt)

                hist_xdata_fine = np.linspace(hist_xdata[0], hist_xdata[-1], 200, endpoint=True)


                mistake_arr = []
                idx_voidpeak = np.argmax(hist[:idx_guess])
                idx_atompeak = np.argmax(hist[idx_guess:])
                t_arr = np.subtract(np.linspace(hist_xdata[idx_voidpeak + int((idx_guess-idx_voidpeak)/2)], hist_xdata[idx_guess + int((idx_atompeak-idx_guess)/2)], 50), hist_xdata[0])
                t2 = time.time()
                for t in t_arr:

                    av_mistake = integrate.quad(gaussian_skew, -100, t, args=(popt[4], popt[5], popt[6], popt[7], 0,), epsabs=1e-1)[0]
                    va_mistake = integrate.quad(emccd_bkg, t, 1e4, args=(popt[0], popt[1], popt[2],popt[3],), epsabs=1e-1)[0]
                    # print(atom_mistake, void_mistake)
                    mistake_arr.append([va_mistake, av_mistake])
                t3 = time.time()
                print(t3-t2)

                threshold_idx = np.argmin(np.sum(mistake_arr, axis=1))
                threshold = t_arr[threshold_idx]+hist_xdata[0]
                #threshold = t_arr[np.argmin(mistake_arr)]
                min_mistake = mistake_arr[threshold_idx]

                tot_area_a = integrate.quad(gaussian_skew, -np.inf, np.inf, args=(popt[4], popt[5], popt[6], popt[7], 0,))[0]
                tot_area_v = integrate.quad(emccd_bkg, -np.inf, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]

                va_mistake, av_mistake = mistake_arr[threshold_idx]
                inf_all.append([va_mistake/tot_area_v, av_mistake/tot_area_a])

            except:
                print('threshold fit failed, using threshold guess')
                fit_worked = False
                threshold = threshold_guess
                inf_all.append([0, 0])

        elif (mode=='emccd_skew_fast'):
            idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

            pguess = np.array([ 1.7, 0,  0.017,
                                np.max(hist[:idx_guess]),  np.max(hist[idx_guess:]), hist_xdata[idx_guess+np.argmax(hist[idx_guess:])]-hist_xdata[0],
                                (hist_xdata[-1]-hist_xdata[idx_guess])/5, 0])
            # try:
            popt, pcov = curve_fit(emccd_hist_skew, np.subtract(hist_xdata, hist_xdata[0]), hist, p0=pguess, bounds=([1, -np.inf, 1e-10, 0, 0, -np.inf, 0, -np.inf], [100, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
            popt_all.append(popt)

            hist_xdata_fine = np.linspace(hist_xdata[0], hist_xdata[-1], 200, endpoint=True)


            mistake_arr = []
            idx_voidpeak = np.argmax(hist[:idx_guess])
            idx_atompeak = np.argmax(hist[idx_guess:])
            t_arr = arr(np.subtract(np.arange(hist_xdata[idx_voidpeak], hist_xdata[idx_atompeak]), hist_xdata[0]), dtype=int)
            t_all = arr(np.arange(t_arr[0]-200, 1e4), dtype=int)
            atom_curve = gaussian_skew(t_all, popt[4], popt[5], popt[6], popt[7], 0)
            void_curve = emccd_bkg(t_all, popt[0], popt[1], popt[2], popt[3])
            t2 = time.time()
            for t in t_arr:
                tidx = t - (t_arr[0]-200)
                atom_mistake = np.sum(atom_curve[:tidx+1])
                void_mistake = np.sum(void_curve[tidx:])
                # print(atom_mistake, void_mistake)
                mistake = void_mistake + atom_mistake
                mistake_arr.append(mistake)
            t3 = time.time()
            print(t3-t2)


            threshold = t_arr[np.argmin(mistake_arr)] + hist_xdata[0]
            #threshold = t_arr[np.argmin(mistake_arr)]
            min_mistake = mistake_arr[np.argmin(mistake_arr)]

            tot_area = np.sum(atom_curve) + np.sum(void_curve)#integrate.quad(gaussian_skew, -np.inf, np.inf, args=(popt[4], popt[5], popt[6], popt[7], 0,))[0] + integrate.quad(emccd_bkg, -np.inf, np.inf, args=(popt[0], popt[1], popt[2],popt[3],))[0]

            infidelity = min_mistake/tot_area
            inf_all.append(infidelity)

            # except:
            #     print('threshold fit failed, using threshold guess')
            #     fit_worked = False
            #     threshold = threshold_guess
            #     min_mistake = 1000

        elif (mode=='void_atom'):
            single_threshold=True
            if num == 0:
                idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

                mistake_arr = []

                t_arr = np.linspace(hist_xdata[np.argmax(hist[:idx_guess])+10], hist_xdata[np.argmax(hist[idx_guess:])-10], 20)
                for t in t_arr:

                    data = get_loss(exp, run, masks, [t]*num_img, crop=crop, output=False, keep_img=[0,1])

                    mistake = data[0]
                    mistake_arr.append(mistake)


                popt, pcov = curve_fit(square, t_arr, mistake_arr, p0=[t_arr[10], 1e-6, 0.1])
                threshold = popt[0]
                min_mistake = popt[-1]

                infidelity = min_mistake
                inf_all.append(infidelity)
            else:
                threshold = threshold



        else:
            threshold = threshold_guess
            min_mistake = 1000

        if num == 0:
            if single_threshold==True:
                threshold_0 = threshold
            threshold_all.append(threshold)

        if num != 0:
            if single_threshold==True:
                threshold = threshold_0
            threshold_all.append(threshold)


    if output == True:
        fig, ax = plt.subplots(figsize=[6,4])
        alphaarr = np.linspace(1.0,0.5, num_img)
        c_ls = ["m", "c", "tab:orange"]
        for num in keep_img:
            ax.plot(hist_xdata_all[num], hist_all[num], alpha=alphaarr[num], drawstyle='steps-post', color='k', zorder = 0)
            ax.plot([threshold_all[num], threshold_all[num]], [np.min(hist_all[num]), np.max(hist_all[num])], linestyle='--', color=c_ls[num], label=num, zorder = 0)

        ax.set_xlabel('Counts collected')
        ax.set_ylabel('Events')

        if (mode=='emccd' and fit_worked):
            ax.plot(hist_xdata_all[0], emccd_hist(hist_xdata_all[0]-hist_xdata_all[0][0], *popt_all[0]), label='fit')

        elif (mode=='emccd_skew' and fit_worked):
            for num in keep_img:
                ax.plot(hist_xdata_all[num], emccd_hist_skew(hist_xdata_all[num]-hist_xdata_all[0][num], *popt_all[num]), label='fit', linewidth=2, color=c_ls[num])


        if logscale == True:
            ax.set_yscale('log')
            ax.set_ylim(0.7, )

        else:
            ax.set_ylim(0, )
        #ax.set_xlim(4800, xlim)
        plt.legend()
        ax.set_title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.show()

        format_string = "{:.3f}"

        if (mode=='cmos' and fit_worked):
            print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
            print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
            print('bkg peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,2]])
            print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
            print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,0]])
            print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            print('fitted infidelity: ',[format_string.format(number*100) for number in inf_all], 'percent')
            print('loss from infidelity: ',[format_string.format(number*100) for number in loss_all], 'percent')

        elif (mode=='emccd' and fit_worked):
            print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
            print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
            print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,6]])
            print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
            print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            print('fitted infidelity: ',[format_string.format(number*100) for number in inf_all], 'percent')

        elif (mode=='emccd_skew' or mode=='emccd_skew_fast' and fit_worked):
            print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
            print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
            print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,6]])
            print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
            print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            print('va fitted infidelity: ',[format_string.format(number*100) for i,number in enumerate(arr(inf_all)[:,0]) if i in keep_img], 'percent')
            print('av fitted infidelity: ',[format_string.format(number*100) for i,number in enumerate(arr(inf_all)[:,1]) if i in keep_img], 'percent')

        elif (mode=='void_atom'):
            print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all) if i in keep_img]])
            print('fitted infidelity: ',[format_string.format(number*100) for number in inf_all], 'percent')

    if (mode=='cmos' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all, loss_all
    elif (mode=='emccd' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all
    elif (mode=='emccd_skew' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all
    elif (mode=='emccd_skew_fast' and fit_worked):
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css, popt_all, inf_all, t_arr, t_all, atom_curve, void_curve
    else:
        return threshold_all, hist_xdata_all, hist_all, min_mistake, css



def find_multiexperiment_threshold(dataAddress, runs, masks, threshold_guess = 22, bin_width = 1, fit = True, crop = [0,None,0,None], output = True):

    hist_xdata_all = []
    hist_all = []
    threshold_all = []
    popt_all = []

    cs = []
    cut = 10
    diffs = []

    for i, run in enumerate(runs):
        exp = ExpFile(dataAddress / 'Raw Data', run)
        bkg = (np.mean(exp.pics[:, :cut, :-cut]) + np.mean(exp.pics[:, :-cut, -cut:]) + np.mean(exp.pics[:, -cut:, cut:]) + np.mean(exp.pics[:, cut:, :cut]))/4
        sig = exp.pics[::2, crop[0]:crop[1], crop[2]:crop[3]]
        diff = sig-bkg
        if (i==0):
            diffs = diff
        else:
            diffs = np.concatenate((diffs, diff), axis=0)
    diffs = np.array(diffs)
    print(np.shape(diffs))

    cs = np.array(list(map(lambda image:
                            list(map(lambda mask:
                                     np.sum(mask*image),
                                     masks)),
                            diff)))
    cs = cs.flatten()

    hist, bin_edges = np.histogram(cs, bins=np.arange(np.min(cs)-bin_width/2, np.max(cs)+bin_width/2, bin_width))
    #hist_xdata = np.add(bin_edges, (bin_edges[1]-bin_edges[0])/2)[:-1]
    hist_xdata = bin_edges[:-1]

    hist_xdata_all.append(hist_xdata)
    hist_all.append(hist)

    fit_worked = True
    if (fit):
        idx_guess = np.abs(hist_xdata - threshold_guess).argmin()

        pguess = [np.max(hist[:idx_guess])-np.min(hist[:idx_guess]),
                  hist_xdata[np.argmax(hist[:idx_guess])],
                  (hist_xdata[idx_guess]-hist_xdata[0])/5,
                  np.max(hist[idx_guess:])-np.min(hist[idx_guess:]),
                  hist_xdata[idx_guess+np.argmax(hist[idx_guess:])],
                  (hist_xdata[-1]-hist_xdata[idx_guess])/5]
        try:
            popt, pcov = curve_fit(double_gaussian, hist_xdata, hist, p0=pguess) #p[0] * np.exp( -((x- p[1])**2) / (2*(p[2]**2)) ) + p[3] * np.exp( -((x - p[4])**2) / (2*(p[5]**2)) )
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
            infidelity = min_mistake/(popt[0]*popt[2]*np.sqrt(2*np.pi)+popt[3]*popt[5]*np.sqrt(2*np.pi))
            infidelity_loss = erfc(threshold, popt[3], popt[4], popt[5])/(popt[3]*popt[5]*np.sqrt(2*np.pi))

        except:
            print('threshold fit failed, using threshold guess')
            fit_worked = False
            threshold = threshold_guess
            min_mistake = 1000

    else:
        threshold = threshold_guess
        min_mistake = 1000

    threshold_all.append(threshold)


    if output == True:
        fig, ax = plt.subplots(figsize=[6,4])
        plt.plot(hist_xdata_all[0], hist_all[0], alpha=0.9, drawstyle='steps', color='k', zorder = 0)
        plt.plot([threshold_all[0], threshold_all[0]], [np.min(hist_all[0]), np.max(hist_all[0])], 'k--', alpha=0.9, label='even', zorder = 0)
#         plt.plot(hist_xdata_all[0], pguess)
        plt.xlabel('Counts collected')
        plt.ylabel('Events')
        plt.ylim(0, )
        plt.legend()
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.show()

        if (fit and fit_worked):
            print('even bkg peak position: '+'{:.3f}'.format(popt_all[0][1]))
            print('even atom peak position: '+'{:.3f}'.format(popt_all[0][4]))
            print('even bkg peak width: '+'{:.3f}'.format(abs(popt_all[0][2])))
            print('even atom peak width: '+'{:.3f}'.format(abs(popt_all[0][5])))
            print('even thresholds: '+'{:.3f}'.format(threshold_all[0]))
            print('fitted infidelity: '+'{:.3f}'.format(infidelity*100)+' percent')
            print('fitted loss from infidelity: '+'{:.3f}'.format(infidelity_loss*100)+' percent')

    if (fit and fit_worked):
        return threshold_all[0], popt_all, hist_xdata_all, hist_all, min_mistake, infidelity, popt_all[0]
    else:
        return threshold_all[0], hist_xdata_all, hist_all, min_mistake


def get_binarized(exp, run, masks, threshold, crop=[0,None,0,None], mode = 'none'):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    cs_arr = []
    cut = 10

    bkg_arr = []
    for num in range(num_img):
        bkg = (np.mean(exp.pics[num::num_img, :cut, :-cut]) + np.mean(exp.pics[num::num_img, :-cut, -cut:]) + np.mean(exp.pics[num::num_img, -cut:, cut:]) + np.mean(exp.pics[num::num_img, cut:, :cut]))/4
        bkg_arr.append(bkg)

    sig = exp.pics[:, crop[0]:crop[1], crop[2]:crop[3]]

    if mode == 'bkg_subtract':
        diff = [np.subtract(s, bkg_arr[i%num_img]) for i,s in enumerate(sig)]
    else:
        diff = sig

    roisums = np.array(list(map(lambda image:list(map(lambda mask:np.sum(mask*image),masks)),diff)))

    if type(threshold)==float:
        binarized = [np.clip(r, threshold, threshold+1) - threshold for i,r in enumerate(roisums)]
    else:
        binarized = [np.clip(r, threshold[i%num_img], threshold[i%num_img]+1) - threshold[i%num_img] for i,r in enumerate(roisums)]

    return binarized

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

def get_loss3(exp, run, masks, t, sortkey=0, crop=[0,None,0,None], output=True, keep_img=[0,1,2]):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    loaddata = data[keep_img[0]::num_img]
    midddata = data[keep_img[1]::num_img]
    survdata = data[keep_img[2]::num_img]

    if (exp.key.ndim > 1):
        key = exp.key[:, sortkey]
    else:
        key = exp.key

    aaa = 0
    aav = 0
    ava = 0
    avv = 0
    vaa = 0
    vav = 0
    vva = 0
    vvv = 0
    a = 0

    for i in range(len(key)):

        for j in range(exp.reps):
            for m in range(len(masks)):
                atom1 = loaddata[i*exp.reps +j][m]
                atom2 = midddata[i*exp.reps +j][m]
                atom3 = survdata[i*exp.reps +j][m]
                if (atom1 and atom2 and atom3):
                    aaa += 1
                elif (atom1 and atom2 and atom3==0):
                    aav += 1
                elif (atom1 and atom2==0 and atom3):
                    ava += 1
                elif (atom1 and atom2==0 and atom3==0):
                    avv += 1
                elif (atom1==0 and atom2 and atom3):
                    vaa += 1
                elif (atom1==0 and atom2 and atom3==0):
                    vav += 1
                elif (atom1==0 and atom2==0 and atom3):
                    vva += 1
                elif (atom1==0 and atom2==0 and atom3==0):
                    vvv += 1
                if (atom1):
                    a += 1

    npair = aaa+aav+ava+avv+vaa+vav+vva+vvv

    load = a/npair
    load_err = np.sqrt(load*(1-load)/npair)

    if a:
        surv = ava/a
        surv_err = np.sqrt(surv*(1-surv)/a)
    else:
        surv = 0
        surv_err = 0

    if (output):
        print(' ')
        print('total pairs: ' + str(npair))
        print('atom void atom: ' + str(ava))
        print('void void void: ' + str(vvv))
        print('atom void void: ' + str(avv))
        print('atom atom (atom/void)' + str(aaa + aav))
        print('void (at least one atom): ' + str(vaa+vav+vva))
        print('load: ' +  '{:.3f}'.format(load*100)+ ' +- '+ '{:.3f}'.format(load_err*100)+ ' percent')
        print('shelved: ' +  '{:.3f}'.format(surv*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')
        print('unshelved: ' +  '{:.3f}'.format((1-surv)*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')

    return (vaa+vav+vva)/npair, ava, avv, (vaa+vav+vva), vvv, surv, surv_err, data

def getMasksManual(mimg,mimg2 = False, red_x=0,red_y = 0,blue_x=0,blue_y = 0,xoffset=0, yoffset=0, fftN = 2000, N = 10, wmask = 3, supersample = None, mode = 'gauss', FFT = True,
                   peakParams = [10,10], output = True, coords = None, mindist=100, disttozero=[50,100,100], get_mask_centers = False, mod2Dgauss = False, peaknumx = 0, peaknumy = 0):
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

        xpeak, ypeak = fMaxCoord[xsort[peaknumx]], fMaxCoord[ysort[peaknumy]]

    #     print(fMaxCoord)
    #     print(xsort)
    #     print(ysort)
        if output == True:
            plt.rcParams["figure.figsize"] = (20,3)

            plt.imshow(fimgAbs)
            plt.colorbar()
            plt.plot(fMaxCoord[:,1], fMaxCoord[:,0],'g.')
            # plt.plot([xpeak[0], ypeak[0]],[xpeak[1],ypeak[1]],'r.')
            xpeak[1] += red_x
            xpeak[0] += red_y
            ypeak[1] += blue_x
            ypeak[0] += blue_y

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

            pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)+yoffset] for x in px]).reshape((N**2,2))

        elif type(N)==list:
            nsx = np.arange(N[1])
            nsy = np.arange(N[0])

            px = arr([(dx*ind) for ind in nsx])
            py = arr([(dy*ind) for ind in nsy])

            pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX)+xoffset, (2*np.pi-phiy)/(2*np.pi*normY)+yoffset] for x in px]).reshape((N[0]*N[1] ,2))

            if output == True:
                fig, ax = plt.subplots()
                plt.imshow(mimg)
                plt.plot(pts[:,1], pts[:,0], 'r.')

            if mod2Dgauss != False:
                chor = mod2Dgauss[0]
                cver = mod2Dgauss[1]
                ptsfit = []
                for pt in pts:
            #        print(pt)
                    x1, x2, y1, y2 = int(pt[1]-chor), int(pt[1]+chor+1), int(pt[0]-cver), int(pt[0]+cver+1)
                    impt = mimg[y1:y2,x1:x2]
            #         plt.imshow(impt)
                    ptfit = gaussFit2d_rot(impt)

                    if np.sqrt(np.diag(ptfit[2]))[0] <100:
                        ptsfit.append([int(pt[0]-cver)+ptfit[1][2],int(pt[1]-chor)+ptfit[1][1]])
                    else:
                        ptsfit.append(pt)
                pts = np.array(ptsfit)


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

            if mod2Dgauss != False:
                chor = mod2Dgauss[0]
                cver = mod2Dgauss[1]
                ptsfit = []
                for pt in pts:
            #        print(pt)
                    x1, x2, y1, y2 = int(pt[1]-chor), int(pt[1]+chor+1), int(pt[0]-cver), int(pt[0]+cver+1)
                    impt = mimg[y1:y2,x1:x2]
            #         plt.imshow(impt)
                    ptfit = gaussFit2d_rot(impt)

                    if np.sqrt(np.diag(ptfit[2]))[0] <100:
                        ptsfit.append([int(pt[0]-cver)+ptfit[1][2],int(pt[1]-chor)+ptfit[1][1]])
                    else:
                        ptsfit.append(pt)
                pts = np.array(ptsfit)
#             pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N[0]*N[1] ,2))
    if output == True:
        if np.all(mimg2) == False:
            fig, ax = plt.subplots()
            plt.imshow(mimg)
        else:
            fig, ax = plt.subplots()
            plt.imshow(mimg2)
        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:

            plt.plot(pts[:,1],pts[:,0],'r.')

        plt.show()

        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:

            plt.plot(pts[:,1],pts[:,0],'r.')


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

    plt.rcParams["figure.figsize"] = (5,4)

    if (get_mask_centers == True):
        return masks, pts
    else:
        return masks

def gaussFit2d_rot(datc):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
    sig_x_guess = np.abs(x[i]-x[ihalf]) - 0.5
    sig_y_guess = np.abs(y[i]-y[ihalf]) - 0.5
#     print(sig_x_guess,sig_y_guess)
    guess = [datf[i], x[i], y[i], 0, sig_x_guess, sig_y_guess]
    pred_params, uncert_cov = curve_fit(gauss2d, xy, datf, p0=guess, maxfev=100000)

    zpred = gauss2d(xy, *pred_params)
    #print('Predicted params:', pred_params)
#     print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params, uncert_cov

def getMasksGuessPts(mimg, pts, mod2Dgauss=[2.0,2.0], wmask = 3, mode = 'gauss',  output = True, coords = None,
                     get_mask_centers = False):
    """Given an averaged atom image, returns list of masks, where each mask corresponds to the appropriate mask for a single atom."""

    chor = mod2Dgauss[0]
    cver = mod2Dgauss[1]
    ptsfit = []
    for pt in pts:
#        print(pt)
        x1, x2, y1, y2 = int(pt[1]-chor), int(pt[1]+chor+1), int(pt[0]-cver), int(pt[0]+cver+1)
        impt = mimg[y1:y2,x1:x2]
#         plt.imshow(impt)
        ptfit = gaussFit2d_rot(impt)

        if np.sqrt(np.diag(ptfit[2]))[0] <100:
            ptsfit.append([int(pt[0]-cver)+ptfit[1][2],int(pt[1]-chor)+ptfit[1][1]])
        else:
            ptsfit.append(pt)
    pts = np.array(ptsfit)

    if output == True:
        fig, ax = plt.subplots(figsize=[20,20])
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
        fig, ax = plt.subplots(figsize=[20,20])
        plt.imshow(np.sum(masks, axis=0))
        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:
            plt.plot(pts[:,1],pts[:,0],'r.')

        plt.show()

    # plt.rcParams["figure.figsize"] = (5,4)

    if (get_mask_centers == True):
        return masks, pts
    else:
        return masks

def gaussFit2d_rot(datc):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
    sig_x_guess = np.abs(x[i]-x[ihalf]) - 0.5
    sig_y_guess = np.abs(y[i]-y[ihalf]) - 0.5
#     print(sig_x_guess,sig_y_guess)
    guess = [datf[i], x[i], y[i], 0, sig_x_guess, sig_y_guess]
    pred_params, uncert_cov = curve_fit(gauss2d, xy, datf, p0=guess, maxfev=100000)

    zpred = gauss2d(xy, *pred_params)
    #print('Predicted params:', pred_params)
#     print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params, uncert_cov

def get_loss(exp, run, masks, t, sortkey=0, crop=[0,None,0,None], output=True, keep_img=[0,1], mode='none',skipFirst=False,Nvalid=0):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop, mode=mode)
    if (exp.key.ndim > 1):
        key = exp.key[:, sortkey]
    else:
        key = exp.key

    if skipFirst==True:
        key = key[1:]
        num_img = int(exp.pics.shape[0]/exp.reps/len(key))
        data = data[exp.reps*num_img:]
        key_name = exp.key_name
    else:
        num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
        data = data
        key = key
        key_name = exp.key_name
    loaddata = data[keep_img[0]::num_img]
    survdata = data[keep_img[1]::num_img]

    av = 0
    va = 0
    vv = 0
    aa = 0
    a = 0

    for i in range(len(key)):
        for j in np.arange(Nvalid,exp.reps):
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
    load_err = np.sqrt(load*(1-load)/npair)

    if a:
        surv = aa/a
        surv_err = np.sqrt(surv*(1-surv)/a)
    else:
        surv = 0
        surv_err = 0

    if (output):
        print(' ')
        print('total pairs: ' + str(npair))
        print('atom atom: ' + str(aa))
        print('void void: ' + str(vv))
        print('atom void: ' + str(av))
        print('void atom: ' + str(va))
        print('load: ' +  '{:.3f}'.format(load*100)+ ' +- '+ '{:.3f}'.format(load_err*100)+ ' percent')
        print('survival: ' +  '{:.3f}'.format(surv*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')
        print('loss: ' +  '{:.3f}'.format((1-surv)*100)+ ' +- '+ '{:.3f}'.format(surv_err*100)+ ' percent')

    return va/npair, aa, av, va, vv, surv, surv_err, data

def var_scan_loadprob(exp, run, masks, t, fit='none', sortkey=0, crop=[0,None,0,None], fullscale=True, img_idx = 0, mode='emccd', skipFirst=False):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop, mode=mode)

    data = data[img_idx::num_img]

    if (exp.key.ndim > 1):
        key = exp.key[:, sortkey]
    else:
        key = exp.key

    patom = []
    patom_err = []
    for i in range(len(exp.key)):
        p = np.sum(data[i*exp.reps : (i + 1)*exp.reps])/exp.reps/len(masks)
        patom.append(p)
        patom_err.append(np.sqrt(p*(1-p)/exp.reps/len(masks)))


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
        if (fullscale==True):
            im = plt.imshow(patom_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                               aspect=(k1max - k1min)/(k0max - k0min), vmin=0, vmax=1, origin="lower")
        else:
            im = plt.imshow(patom_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                               aspect=(k1max - k1min)/(k0max - k0min), origin="lower")

        plt.xlabel(exp.key_name[1])
        plt.ylabel(exp.key_name[0])
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('load_prob')
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.show()



    else:
        patom_sorted = np.array(patom)[np.argsort(key)]
        patom_err_sorted = np.array(patom_err)[np.argsort(key)]

        key_sorted = np.sort(key)

        if fit == 'gaussian_peak':
            try:
                pguess = [np.max(patom_sorted)-np.min(patom_sorted),
                          key_sorted[np.argmax(patom_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.min(patom_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)

                print('key fit = {:.3e}'.format(popt[1]))
            except:
                print('fit failed')

        if fit == 'gaussian_dip':
            pguess = [-np.max(patom_sorted)+np.min(patom_sorted),
                      key_sorted[np.argmin(patom_sorted)],
                      (key_sorted[-1]-key_sorted[0])/3,
                      np.max(patom_sorted)]
            popt, pcov = curve_fit(gaussian, key_sorted, patom_sorted, p0=pguess)


            print('a, x0, sig, y0')
            print(popt)

            print('key fit = {:.3e}'.format(popt[1]))


        if fit == 'hockey_fall':
            pguess = [(key_sorted[-1]+key_sorted[0])/2,
                      (np.max(patom_sorted)-np.min(patom_sorted))/ (key_sorted[-1]-key_sorted[0]),
                      np.max(patom_sorted)]
            popt, pcov = curve_fit(hockey_fall, key_sorted, patom_sorted, p0=pguess)

            print('key fit = {:.3e}'.format(popt[0]))

        if fit == 'hockey_rise':
            pguess = [(key_sorted[-1]+key_sorted[0])/2,
                      (np.max(patom_sorted)-np.min(patom_sorted))/ (key_sorted[-1]-key_sorted[0]),
                      np.max(patom_sorted)]
            popt, pcov = curve_fit(hockey_rise, key_sorted, patom_sorted, p0=pguess)

            print('key fit = {:.3e}'.format(popt[0]))

        # patom_uncertainty = np.sqrt(exp.reps*len(masks)*patom_sorted)/(exp.reps*len(masks))

        key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
        fig, ax = plt.subplots(figsize=[5,4])
        if fit == 'gaussian_peak' or fit == 'gaussian_dip':
            plt.plot(key_fine, gaussian(key_fine, *popt), 'k-')
        if fit == 'hockey_fall':
            plt.plot(key_fine, hockey_fall(key_fine, *popt), 'k-')
        if fit == 'hockey_rise':
            plt.plot(key_fine, hockey_rise(key_fine, *popt), 'k-')
        plt.errorbar(key_sorted, patom_sorted, patom_err_sorted, color='k', marker='o', linestyle=':', alpha=0.7)
        if (type(exp.key_name) == str):
            plt.xlabel(exp.key_name)
        else:
            plt.xlabel(exp.key_name[sortkey])
        plt.ylabel('load prob')
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        if (fullscale==True):
            plt.ylim(0, 1)

        plt.show()

    if (exp.key.ndim > 1):
        return patom_sorted
    else:
        return key_sorted, patom_sorted

def var_scan_survprob(exp, run, masks, t, fit='none', sortkey=[0,1], crop=[0,None,0,None], pguess=None, multiScan=False, fullscale=True, plot=True, keep_img=[0,1], mode='emccd', skip=False,
                      postselection = False,parity = False, skipFirst=False, skipReps = None, order ='2',cutnum = False):




    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop, mode=mode)
    if skipFirst==True:
        key = exp.key[1:]
        num_img = int(exp.pics.shape[0]/exp.reps/len(key))
        data = data[exp.reps*num_img:]
        key_name = exp.key_name
    else:
        num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
        data = data
        key = exp.key
        key_name = exp.key_name

    reps = exp.reps
    if cutnum != False:
        print("cutnum = {:.2f}".format(cutnum))
        if cutnum >= 0:
            datas = []
            for index in range(arr(data).shape[0]):
                if index%(num_img*exp.reps) >= cutnum*num_img:
                    datas.append(data[index])
            data = np.array(datas)
            reps = exp.reps-cutnum
        elif cutnum < 0:
            datas = []
            for index in range(arr(data).shape[0]):
                if index%(num_img*exp.reps) < (exp.reps+cutnum+1)*num_img:
                    datas.append(data[index])
            data = np.array(datas)
            reps = exp.reps+cutnum+1


    loaddata = data[keep_img[0]::num_img]
    return_loaddata = loaddata
    survdata = data[keep_img[1]::num_img]

    if postselection != False:
        if order == '1':
            loaddata1 = arr(loaddata)[:,:masks.shape[0]//2]
            loaddata2 = arr(loaddata)[:,masks.shape[0]//2:masks.shape[0]]
            survdata1 = arr(survdata)[:,:masks.shape[0]//2]
            survdata2 = arr(survdata)[:,masks.shape[0]//2:masks.shape[0]]
        elif order == '2':
            loaddata1 = arr(loaddata)[:,::2]
            loaddata2 = arr(loaddata)[:,1::2]
            survdata1 = arr(survdata)[:,::2]
            survdata2 = arr(survdata)[:,1::2]


        dimer = loaddata1*loaddata2

        if postselection == 'dimer':
            if parity == True:
                print('Probablity of parity even')
                loaddata = np.concatenate((dimer,np.zeros(dimer.shape)),axis=1)
                survdata = np.concatenate(((survdata1*dimer + survdata2*dimer + 1)%2,np.zeros(dimer.shape)),axis=1)
            elif parity == '11':
                print('Probablity of 11')
                loaddata = np.concatenate((dimer,np.zeros(dimer.shape)),axis=1)
                dimer_sum = survdata1*dimer + survdata2*dimer
                survdata = np.concatenate((np.where(dimer_sum>1,dimer_sum, 0)/2,np.zeros(dimer.shape)),axis=1)
            elif parity == '00':
                print('Probablity of 00')
                loaddata = np.concatenate((dimer,np.zeros(dimer.shape)),axis=1)
                dimer_sum = survdata1*dimer + survdata2*dimer
                survdata = np.concatenate((np.where((2-dimer_sum)>1,(2-dimer_sum), 0)/2*dimer,np.zeros(dimer.shape)),axis=1)
            else:
                loaddata = np.concatenate((loaddata1*dimer,loaddata2*dimer),axis=1)
                survdata = np.concatenate((survdata1*dimer,survdata2*dimer),axis=1)
        elif postselection == 'singlet':
            loaddata = np.concatenate((loaddata1*(1-dimer),loaddata2*(1-dimer)),axis=1)
            survdata = np.concatenate((survdata1*(1-dimer),survdata2*(1-dimer)),axis=1)

            return_loaddata = loaddata

    surv_prob = []
    surv_prob_uncertainty = []
    popt, pcov = [], []

    if multiScan == True:
        key = key[:, sortkey]
        print(np.shape(key))
        if (isinstance(sortkey,int)):
            key_name = key_name[sortkey]
        else:
            key_name = [key_name[i] for i in sortkey]
    else:
        key = key
        key_name = key_name


    for i in range(len(key)):
        aa = 0
        a = 0
        for j in range(reps):
            # if (key.shape == (1)):
            for m in range(len(masks)):
                atom1 = loaddata[i*reps +j][m]
                atom2 = survdata[i*reps +j][m]
                if (atom1 and atom2):
                    aa += 1
                if (atom1):
                    a += 1
        if a:
            p = aa/a
            surv_prob.append(p)
            surv_prob_uncertainty.append(np.sqrt(p*(1-p)/a))
        else:
            surv_prob.append(0)
            surv_prob_uncertainty.append(0)

    if (np.shape(key)[-1] == 2):

        surv_prob_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), surv_prob))), key=lambda x: [x[0], x[1]]))
        surv_prob_sorted_reshape = np.reshape(surv_prob_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))
        surv_prob_uncertainty_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key),surv_prob_uncertainty))), key=lambda x: [x[0], x[1]]))
        surv_prob_sorted_reshape_err = np.reshape(surv_prob_uncertainty_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))

        k0min = np.min(surv_prob_sorted[:,0])
        k0max = np.max(surv_prob_sorted[:,0])
        k1min = np.min(surv_prob_sorted[:,1])
        k1max = np.max(surv_prob_sorted[:,1])
        key0 = np.sort(np.unique(surv_prob_sorted[:,0]))
        key1 = np.sort(np.unique(surv_prob_sorted[:,1]))

        if (plot):
            fig, ax = plt.subplots(figsize=[5,4])
            if (fullscale==True):
                im = plt.imshow(surv_prob_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                           aspect=(k1max - k1min)/(k0max - k0min), vmin=0, vmax=1, origin="lower")
            elif (fullscale==False):
                im = plt.imshow(surv_prob_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                           aspect=(k1max - k1min)/(k0max - k0min), origin="lower")
            else:
                im = plt.imshow(surv_prob_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                           aspect=(k1max - k1min)/(k0max - k0min),vmin=fullscale[0], vmax=fullscale[1], origin="lower")
                print('vmin: {} vmax:  {}'.format(fullscale[0],fullscale[1]))
            plt.xlabel(key_name[1])
            plt.ylabel(key_name[0])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('surv_prob')
            plt.title(str(exp.data_addr) + "\data_" + str(run) + ".h5")
            plt.show()

    if (np.shape(key)[-1] == 3):

        surv_prob_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), surv_prob))), key=lambda x: [x[0], x[1], x[2]]))
        surv_prob_sorted_reshape = np.reshape(surv_prob_sorted[:,3], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1])), len(np.unique(key[:, 2]))))
        surv_prob_uncertainty_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key),surv_prob_uncertainty))), key=lambda x: [x[0], x[1], x[2]]))
        surv_prob_sorted_reshape_err = np.reshape(surv_prob_uncertainty_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1])), len(np.unique(key[:, 2]))))

        k0min = np.min(surv_prob_sorted[:,0])
        k0max = np.max(surv_prob_sorted[:,0])
        k1min = np.min(surv_prob_sorted[:,1])
        k1max = np.max(surv_prob_sorted[:,1])
        k2min = np.min(surv_prob_sorted[:,2])
        k2max = np.max(surv_prob_sorted[:,2])
        key0 = np.sort(np.unique(surv_prob_sorted[:,0]))
        key1 = np.sort(np.unique(surv_prob_sorted[:,1]))
        key2 = np.sort(np.unique(surv_prob_sorted[:,2]))

        keylengths = np.array([len(np.unique(key[:, 0])), len(np.unique(key[:, 1])), len(np.unique(key[:, 2]))])
        minnumkey = np.min(keylengths)
        minnumarg = np.argmin(keylengths)

    else:

        surv_prob_sorted = np.array(surv_prob)[np.argsort(key)]
        surv_prob_uncertainty_sorted = np.array(surv_prob_uncertainty)[np.argsort(key)]

        key_sorted = np.sort(key)

        if (key.ndim == 1):
            fitFunc = None
            if fit == 'line':
                pguess = [(np.max(surv_prob_sorted)-np.min(surv_prob_sorted))/(key_sorted[-1]-key_sorted[0]),
                          0]
                popt, pcov = curve_fit(line, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = line
                print('a, b')
                print(popt)
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[0], np.sqrt(np.diag(pcov))[0]))



            if fit == 'gaussian_peak':
                pguess = [np.max(surv_prob_sorted)-np.min(surv_prob_sorted),
                          key_sorted[np.argmax(surv_prob_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = gaussian
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[1], np.sqrt(np.diag(pcov))[1]))
                print(popt)

            if fit == 'gaussian_dip':
                pguess = [-np.max(surv_prob_sorted)+np.min(surv_prob_sorted),
                          key_sorted[np.argmin(surv_prob_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = gaussian
                print('a, x0, sig, y0')
                print(popt)
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[1], np.sqrt(np.diag(pcov))[1]))

            if fit == 'doublegaussian_dip':
                if (pguess == None):
                    pguess = [-np.max(surv_prob_sorted)+np.min(surv_prob_sorted),-np.max(surv_prob_sorted)+np.min(surv_prob_sorted),
                              key_sorted[np.argmin(surv_prob_sorted)],key_sorted[np.argmin(surv_prob_sorted)],
                              (key_sorted[-1]-key_sorted[0])/5,(key_sorted[-1]-key_sorted[0])/5,
                              np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(twogaussian, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = twogaussian
                print("x0, x1, a0, a1, sig0, sig1, y0")
                print(popt)
                print(np.sqrt(np.diag(pcov)))
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[0], np.sqrt(np.diag(pcov))[0]))
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[1], np.sqrt(np.diag(pcov))[1]))

            if fit == 'hockey_fall':
                pguess = [(key_sorted[-1]+key_sorted[0])/2,
                          (np.max(surv_prob_sorted)-np.min(surv_prob_sorted))/ (key_sorted[-1]-key_sorted[0]),
                          np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(hockey_fall, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = hockey_fall
                print('key fit = {:.3e}'.format(popt[0]))

            if fit == 'hockey_rise':
                pguess = [(key_sorted[-1]+key_sorted[0])/2,
                          (np.max(surv_prob_sorted)-np.min(surv_prob_sorted))/ (key_sorted[-1]-key_sorted[0]),
                          np.max(surv_prob_sorted)]
                popt, pcov = curve_fit(hockey_rise, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = hockey_rise
                print('key fit = {:.3e}'.format(popt[0]))

            if fit == 'triplor':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                if (pguess == None):
                    pguess = [0.4, 0.4,0.4, 0.01, 0.01,
                          (key_sorted[-1]+key_sorted[0])/2,
                          0.14,0]
                popt, pcov = curve_fit(triplor, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = triplor
                print('a0, a1, a2, kc, ks, x0, dx, y0')
                print(popt)
                print('red sideband freq:', popt[5]-abs(popt[6]))

            if fit == 'twogaussian':
                # twogaussian(x, x0, x1, a0, a1, sig0, sig1, y0)
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/3, 2*(key_sorted[-1]+key_sorted[0])/3,
                              0.4, 0.4, 0.01, 0.01, 0]
                popt, pcov = curve_fit(twogaussian, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = twogaussian
                print('x0, x1, a0, a1, sig0, sig1, y0')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))
                print('nbar: %.3f' %((popt[3]/popt[2])/(1-popt[3]/popt[2])))
                print('trap freq: %.3f kHz' %((popt[1]-popt[0])/2))

            if fit == 'tripgaus':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                if (pguess == None):
                    pguess = [0.4, 0.4,0.4, 0.01, 0.01,
                          (key_sorted[-1]+key_sorted[0])/2,
                          0.14,0]
                popt, pcov = curve_fit(tripgaus, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = tripgaus
                print('a0, a1, a2, sc, ss, x0, dx, y0')
                print(popt)
                print('red sideband freq:', popt[5]+abs(popt[6]))

            if fit == 'tripgaus_letting_the_peaks_roam_free':
                # tripgaus_letting_the_peaks_roam_free(x, a0, a1, a2, sc, ss, x0, x1, x2, y0)
                if (pguess == None):
                    pguess = [0.4, 0.4,0.4, 0.01, 0.01,
                          (key_sorted[-1]+key_sorted[0])/2-0.14,
                          (key_sorted[-1]+key_sorted[0])/2,
                          (key_sorted[-1]+key_sorted[0])/2+0.14,
                          0]
                popt, pcov = curve_fit(tripgaus_letting_the_peaks_roam_free, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = tripgaus_letting_the_peaks_roam_free
                print('a0, a1, a2, sc, ss, x0, x1, x2, y0')
                print(popt)

            if fit == 'quadgaus':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                if (pguess == None):
                    pguess = [0.4, 0.4,0.4, 0.4, 0.01, 0.01, 0.01,
                          (key_sorted[-1]+key_sorted[0])/2,
                          0.14, 0.14, 99.86, 0]
                popt, pcov = curve_fit(quadgaus, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = quadgaus
                print('a0, a1, a2, a3, sc, ss, s3, x0, dx1, dx2, x3, y0')
                print(popt)
                print('red sideband freq:', popt[5]-abs(popt[6]))

            if fit == 'quintgaus':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                if (pguess == None):
                    pguess = [0.4, 0.4,0.4, 0.4, 0.4, 0.01, 0.01, 0.01,
                          (key_sorted[-1]+key_sorted[0])/2,
                          0.14, 0.14, 99.86, 0]
                popt, pcov = curve_fit(quintgaus, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = quintgaus
                print('a0, a1, a2, a3, a4, sc, ss, s3, x0, dx1, dx2, dx3, y0')
                print(popt)
                print('red sideband freq:', popt[5]-abs(popt[6]))

            if fit == 'ramanRabi_pol':
                #np.abs(A*np.sin((x-x0)/T)*np.cos(theta)*np.sqrt(np.cos((x-x0)/T)**2+(np.sin((x-x0)/T)*np.sin(theta))**2))
                if (pguess == None):
                    pguess = [0, np.pi()/4/360, 600, 0]
                popt, pcov = curve_fit(ramanRabi_pol, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = ramanRabi_pol
                print('theta, T, a0, x0')
                print(popt)

            if fit == 'sinA1':
                if (pguess == None):
                    pguess = [1, 0, 0.5]
                popt, pcov = curve_fit(sinA1, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = sinA1
                print('f', 'x0', 'y0')
                print(popt)

            if fit == 'sinc2':
                # A*np.sinc(x-x0) + y0
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/2, 0.8, 0, 0.01]
                popt, pcov = curve_fit(sinc2, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = sinc2
                print('x0, a0, y0, k0')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))


            if fit == 'rabi_pi':
                # a0*Omega**2/(Omega**2+(d-d0)**2)*np.sin(np.sqrt((d-d0)**2+Omega**2)*2*3.14159265/Omega/2)**2 + y0
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/2, 1, 0, (key_sorted[-1]-key_sorted[0])/2]
                popt, pcov = curve_fit(rabi_pi, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = rabi_pi
                print('d0, a0, y0, Omega')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'ramsey':
                # a0*Omega**2/(Omega**2+(d-d0)**2)*np.sin(np.sqrt((d-d0)**2+Omega**2)*2*3.14159265/Omega/2)**2 + y0
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/2, 1, 0, (key_sorted[-1]-key_sorted[0])/2, 0.001, 0.01]
                popt, pcov = curve_fit(ramsey, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = ramsey
                print('d, d0, a0, y0, Omega0, T, tau')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'twosinc2':
                # A*np.sinc(x-x0) + y0
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/3,2*(key_sorted[-1]+key_sorted[0])/3, 0.8, 0.8, 0, 0.01, 0.01]
                popt, pcov = curve_fit(twosinc2, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = twosinc2
                print('x0, x1, a0, a1, y0, k0, k1')
                print(popt)

            if fit == 'lor':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                pguess = [0.4, 0.05,
                          (key_sorted[-1]+key_sorted[0])/2
                        ]
                popt, pcov = curve_fit(lor, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = lor
                print('a, k, x0')

                print(popt)

            if fit == 'twolor':
                # twolor(x, a0, a1, k0, k1, x0, x1, y0):
                if (pguess == None):
                    pguess = [0.4, 0.4, 0.05, 0.05,
                              (key_sorted[-1]+key_sorted[0])/3, (key_sorted[-1]+key_sorted[0])*2/3, 0
                            ]
                popt, pcov = curve_fit(twolor, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = twolor
                print('a0, a1, k0, k1, x0, x1, y0')
                print(popt)

            if fit == 'dampedCos':
                # dampedCos(t, A, tau, f, phi, y0): A*np.exp(-t/tau)/2 * (np.cos(2*np.pi*f*t+phi)) + y0
                if (pguess == None):
                    pguess = [np.max(surv_prob_sorted)-np.min(surv_prob_sorted),
                              key_sorted[-1]/2, 4/(key_sorted[-1]), 0,
                              0]
                popt, pcov = curve_fit(dampedCos, key_sorted, surv_prob_sorted, p0=pguess, bounds=(0,[1, 1e6, 1e9, 2*np.pi, 0.8]))
                fitFunc = dampedCos
                print(' A, tau, f, phi, y0')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'cos':
                if (pguess == None):
                    f = 2/(key_sorted[-1]-key_sorted[0])
                    A = np.max(surv_prob_sorted)-np.min(surv_prob_sorted)
                    y0 = 0.5
                    phi = np.arccos(2*(surv_prob_sorted[0]-y0)/A) - 2*np.pi*f*key_sorted[0]
                    pguess = [f, A, phi, y0]
                popt, pcov = curve_fit(cos, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = cos
                print('f, A, phi, y0')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'cosRam':
                # dampedCos(t, A, tau, f, phi, y0): A*np.exp(-t/tau)/2 * (np.cos(2*np.pi*f*t+phi)) + y0
                if (pguess == None):
                    pguess = [1/((key_sorted[-1]-key_sorted[-1])), np.max(surv_prob_sorted)-np.min(surv_prob_sorted), 0, 0.5]
                popt, pcov = curve_fit(cosRam, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = cosRam
                print('f, yup, phi, ydown')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'gausCos':
                #gausCos(t, A, sig, f, phi, y0):(A/2)*np.exp(-(t/sig)**2/2) * (np.cos(2*np.pi*f*t+phi)) + y0
                if (pguess == None):
                    pguess = [np.max(surv_prob_sorted)-np.min(surv_prob_sorted),
                              key_sorted[-1]/2, 4/(key_sorted[-1]), 0,
                              (np.max(surv_prob_sorted)-np.min(surv_prob_sorted))/2]
                popt, pcov = curve_fit(gausCos, key_sorted, surv_prob_sorted, p0=pguess, bounds=(0,[1, 1e6, 1e9, 2*np.pi, 1]))
                sig = np.sqrt(2)*popt[1]
                fitFunc = gausCos
                print(' A, sig, f, phi, y0')

                print(popt)

            if fit == 'Thermal_dephase':
                #Thermal_dephase(t, nbar, Omega0, A, eta=0.33)
                pguess = [1, 4/(key_sorted[-1]), 1]
                popt, pcov = curve_fit(Thermal_dephase, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = Thermal_dephase
                print('nbar, Omega0, A, eta=0.33')
                print(popt)

            if fit == 'decayt':
                # decayt(t, tau, a, y0): y0 + a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(surv_prob_sorted)-np.min(surv_prob_sorted), np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(decayt, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = decayt
                print('tau, a, y0 ', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt_gaussian':
                # decayt(t, tau, a, y0): y0 + a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(surv_prob_sorted)-np.min(surv_prob_sorted), np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(decayt_gaussian, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = decayt_gaussian
                print('tau, a, y0 ', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt_gaussian0':
                # decayt(t, tau, a, y0): y0 + a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(surv_prob_sorted)-np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(decayt_gaussian0, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = decayt_gaussian0
                print('tau, a ', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt0':
                # decayt0(t, tau, a): a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(surv_prob_sorted)-np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(decayt0, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = decayt0
                print('t, tau, a', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt1':
                # decayt1(t, tau, a): a*np.exp(-t/tau)+1
                pguess = [key_sorted[-1]/5, np.max(surv_prob_sorted)-np.min(surv_prob_sorted)]
                popt, pcov = curve_fit(decayt1, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = decayt1
                print('t, tau, a', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depol':
                # depol(x,traman,tau): (1/2*exp(-x/tau))*(1-exp(-2*x*gamma))
                pguess = [1000, 5000]
                popt, pcov = curve_fit(depol, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depol
                print('x, traman, tau', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depolknowntau':
                # depolknowntau(x,traman): (1/2*exp(-x/5687))*(1-exp(-2*x*gamma))
                pguess = [1000]
                popt, pcov = curve_fit(depolknowntau, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depolknowntau
                print('x, traman', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depolimpureknowntau':
                # depolimpureknowntau(x,traman,prob): (1/2*np.exp(-(x+30)*(1/6070+2/traman)))*(-1+2*.07+np.exp(2*(x+30)/traman))
                pguess = [1000,0.05]
                popt, pcov = curve_fit(depolimpureknowntau, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depolimpureknowntau
                print('x, traman, p', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depoloppknowntau':
                pguess = [1000]
                popt, pcov = curve_fit(depoloppknowntau, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depoloppknowntau
                print('x, traman', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depolheuristicloss':
                # depolknowntau(x,traman): (1/2*exp(-x/5687))*(1-exp(-2*x*gamma))
                pguess = [1000]
                popt, pcov = curve_fit(depolheuristicloss, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depolheuristicloss
                print('x, t1', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depoloppheuristicloss':
                # depolknowntau(x,traman): (1/2*exp(-x/5687))*(1-exp(-2*x*gamma))
                pguess = [1000]
                popt, pcov = curve_fit(depoloppheuristicloss, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depoloppheuristicloss
                print('x, t1', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'depolimpureheuristicloss':
                # depolknowntau(x,traman): (1/2*exp(-x/5687))*(1-exp(-2*x*gamma))
                pguess = [1000,0.01]
                popt, pcov = curve_fit(depolimpureheuristicloss, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = depolimpureheuristicloss
                print('x, t1, p', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'expatomloss':
                # depolknowntau(x,traman): (1/2*exp(-x/5687))*(1-exp(-2*x*gamma))
                pguess = [.0001,.000000,1]
                popt, pcov = curve_fit(expatomloss, key_sorted, surv_prob_sorted, p0=pguess)
                fitFunc = expatomloss
                print('x, a, b, Amp', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if (plot):
                key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
                fig, ax = plt.subplots(figsize=(5,4))
                if (fitFunc != None):
                    plt.plot(key_fine, fitFunc(key_fine, *popt), 'k-')
                plt.errorbar(key_sorted, surv_prob_sorted, surv_prob_uncertainty_sorted, color='k', marker='o', linestyle=':', alpha=0.7)
                plt.xlabel(key_name)
                plt.ylabel('surv prob')
                plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
                if (fullscale==True):
                    plt.ylim(0, 1)
                elif (fullscale==False):
                    print('false scale')
                else:
                    plt.ylim(fullscale[0],fullscale[1])
                    print('ylim: {}'.format(fullscale))
                plt.show()

            if (fitFunc != None):
                Nexp = fitFunc(key_sorted, *popt)
                r = surv_prob_sorted - Nexp
                RSS = np.divide(r,surv_prob_uncertainty_sorted) ** 2
                chisq = np.sum(RSS)
                redchisq = chisq/(len(surv_prob_sorted)-len(pguess)+1)
                print('red chisq', redchisq)

    if (np.shape(key)[-1] == 2):
        return surv_prob_sorted_reshape, surv_prob_sorted_reshape_err, surv_prob, key0, key1
    if (np.shape(key)[-1] == 3):
        return surv_prob_sorted_reshape, surv_prob_sorted_reshape_err, surv_prob, key0, key1, key2
    else:
        return key_sorted, surv_prob_sorted, surv_prob_uncertainty_sorted, popt, pcov, return_loaddata


def get_multiexperiment_survival(dataAddress, runs, masks, tfixed, crop=[0,None,0,None],Nvalid=0,scripttype = "master"):
    aas = []
    avs = []
    vas = []
    vvs = []
    s = []
    serr = []
    scriptnames = []
    #t = find_multiexperiment_threshold(dataAddress, runs[::2], masks, threshold_guess = tguess, bin_width = 1, fit=True, crop=crop, output=True)


    for run in runs:
        exp = ExpFile(dataAddress / 'Raw Data', run)

        vanpair, aa, av, va, vv, surv, surv_err, data = get_loss(exp, run, masks, t=tfixed, sortkey=0, crop=crop, output=False,Nvalid=Nvalid)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        if scripttype=='awg':
            exp.print_awg_script_name()
            output = os.path.basename(new_stdout.getvalue())
        else:
            exp.print_master_script_name()
            output = new_stdout.getvalue()
        scriptnames.append(output)
        sys.stdout = old_stdout
        aas.append(aa)
        avs.append(av)
        vas.append(va)
        vvs.append(vv)
        s.append(surv)
        serr.append(surv_err)

    aas = np.array(aas)
    avs = np.array(avs)
    vas = np.array(vas)
    vvs = np.array(vvs)
    s = np.array(s)
    serr = np.array(serr)
    return aas, avs, vas, vvs, s, serr, scriptnames


def get_site_survival(exp, run, masks, t, crop=[0,None,0,None], size=[5,5], output=True,fullscale=True, keep_img=[0,1], mode='emccd'):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    loaddata = data[keep_img[0]::num_img]
    survdata = data[keep_img[1]::num_img]

    surv_probs = []
    surv_prob_uncertaintys = []

    key = exp.key

    for i in range(len(key)):
        surv_prob = []
        surv_prob_uncertainty = []
        for m in range(len(masks)):
            aa = 0
            a = 0
            for j in range(exp.reps):
                atom1 = loaddata[i*exp.reps +j][m]
                atom2 = survdata[i*exp.reps +j][m]
                if (atom1 and atom2):
                    aa += 1
                if (atom1):
                    a += 1
            if np.sum(a):
                p = np.sum(aa)/np.sum(a)
                surv_prob.append(p)
                if np.sum(aa):
                    surv_prob_uncertainty.append(np.sqrt(p*(1-p)/a))
                else:
                    surv_prob_uncertainty.append(0)
            else:
                surv_prob.append(0)
                surv_prob_uncertainty.append(0)

        surv_probs.append(surv_prob)
        surv_prob_uncertaintys.append(surv_prob_uncertainty)

    key_sorted = np.sort(key)
    surv_probs_sorted = np.array(surv_probs)[np.argsort(key),:]
    surv_prob_uncertaintys_sorted = np.array(surv_prob_uncertaintys)[np.argsort(key),:]

    if output==True:
        if fullscale == True:
            plt.figure()
            plt.imshow(surv_probs_sorted[0].reshape(size[0],size[1]), origin='lower', vmin=0, vmax=1)
            plt.colorbar()
        else:
            plt.figure()
            plt.imshow(surv_probs_sorted[0].reshape(size[0],size[1]), origin='lower', vmin=np.min(surv_probs_sorted[0]), vmax=np.max(surv_probs_sorted[0]))
            plt.colorbar()

    return surv_probs_sorted[0], surv_prob_uncertaintys_sorted[0]

def get_rep_survival(exp, run, masks, t, crop=[0,None,0,None], window=100, keep_img=[0,1], plot=True):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    loaddata = data[keep_img[0]::num_img]
    survdata = data[keep_img[1]::num_img]

    surv_probs = []
    surv_prob_uncertaintys = []

    key = exp.key

    for i in range(len(key)):
        surv_prob = []
        surv_prob_uncertainty = []
        noloss = []
        badloss = []
        for j in range(exp.reps):
            aa = 0
            a = 0
            for m in range(len(masks)):
                atom1 = loaddata[i*exp.reps +j][m]
                atom2 = survdata[i*exp.reps +j][m]
                if (atom1 and atom2):
                    aa += 1
                if (atom1):
                    a += 1
            if a:
                surv_prob.append(aa/a)

        if plot:
            fig, ax = plt.subplots(figsize=(5,1), dpi=144)
            n = window
            c=len(surv_prob)-n

            B0survive=[]
            B0errsurvive = []
            N = np.linspace(0,len(surv_prob),c)
            for i in range(c):
                B0survive.append(np.mean(surv_prob[i:i+n]))
                B0errsurvive.append(np.std(surv_prob[i:i+n])/np.sqrt(n))

            ax.set_title('n={} window average survival'.format(n))

            #ax.errorbar(N,B0,B0err)
            plt.plot(N,B0survive,"r")
            plt.show()

            fig, ax = plt.subplots(figsize=(5,1), dpi=144)

            #ax.errorbar(N,B0,B0err)
            plt.plot(N,B0errsurvive,"b")
            plt.title('std')
            plt.show()
    return B0survive, B0errsurvive



def get_site_load(exp, run, masks, t, crop=[0,None,0,None], size=[5, 5],output=True, img_idx = 0, keep_img=[0,1],fullscale = True):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    # data = data[img_idx::num_img]
    loaddata = data[keep_img[0]::num_img]
    #survdata = data[keep_img[1]::num_img]

    load_probs = []
    load_prob_uncertaintys = []

    key = exp.key

    for i in range(len(key)):
        load_prob = []
        load_prob_uncertainty = []
        for m in range(len(masks)):
            a = 0
            for j in range(exp.reps):
                atom1 = loaddata[i*exp.reps +j][m]
                if (atom1):
                    a += 1
            if np.sum(a):
                p = np.sum(a)/exp.reps
                load_prob.append(p)
                if np.sum(a):
                    load_prob_uncertainty.append(np.sqrt(p*(1-p)/exp.reps))
                else:
                    load_prob_uncertainty.append(0)
            else:
                load_prob.append(0)
                load_prob_uncertainty.append(0)

        load_probs.append(load_prob)
        load_prob_uncertaintys.append(load_prob_uncertainty)

    key_sorted = np.sort(key)
    load_probs_sorted = np.array(load_probs)[np.argsort(key),:]
    load_prob_uncertaintys_sorted = np.array(load_prob_uncertaintys)[np.argsort(key),:]


    if output==True:
        if fullscale == True:
            plt.figure()
            plt.imshow(load_probs_sorted[0].reshape(size[0],size[1]), origin='lower', vmin=0, vmax=1)
            plt.colorbar()
        else:
            plt.figure()
            plt.imshow(load_probs_sorted[0].reshape(size[0],size[1]), origin='lower', vmin=np.min(load_probs_sorted[0]), vmax=np.max(load_probs_sorted[0]))
            plt.colorbar()

    return key_sorted, load_probs_sorted, load_prob_uncertaintys_sorted


def get_rep_load(exp, run, masks, t, crop=[0,None,0,None], mode = 'good', img_idx = 0):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))
    data = get_binarized(exp, run, masks=masks, threshold=t, crop=crop)
    data = data[img_idx::num_img]

    load_probs = []
    load_prob_uncertaintys = []

    key = exp.key

    for i in range(len(key)):
        load_prob = []
        load_prob_uncertainty = []
        goodload = []
        badload = []

        for j in range(exp.reps):
            a = 0
            for m in range(len(masks)):
                atom1 = data[i*exp.reps +j][m]
                if (atom1):
                    a += 1
            if a:
                p = a/len(masks)
                load_prob.append(p)
                if np.sum(a):
                    load_prob_uncertainty.append(np.sqrt(p*(1-p)/len(masks)))
                else:
                    load_prob_uncertainty.append(0)
                if p <0.8:
                    badload.append(j)

                if p == 0.99:
                    goodload.append(j)

                #if p == 1:
                #    print('100 percent load for rep:', j)
            else:
                load_prob.append(0)
                load_prob_uncertainty.append(0)

        if mode == 'good':
            print('99 percent load for rep:', goodload)
        if mode == 'bad':
            print('bad load for rep:', badload)
        load_probs.append(load_prob)
        load_prob_uncertaintys.append(load_prob_uncertainty)

    key_sorted = np.sort(key)
    load_probs_sorted = np.array(load_probs)[np.argsort(key),:]
    load_prob_uncertaintys_sorted = np.array(load_prob_uncertaintys)[np.argsort(key),:]

    return key_sorted, load_probs_sorted, load_prob_uncertaintys_sorted


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
    plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")

    return key_sorted, cs_arr_cond_mean_sorted

def get_blowaway_error(tb, tbp, t0, t1, mode='exp'):
    t = np.linspace(t0,t1,100)
    if mode=='exp':
        errsum = np.exp(-t/tb) + (1-np.exp(-t/tbp))
    elif mode=='gaus':
        errsum = np.exp(-t**2/tb**2/2) + (1-np.exp(-t/tbp))
    plt.plot(t, errsum, 'k-')
    plt.xlabel('blowaway time (ms)')
    plt.ylabel('pumping + survival error')
    return t[np.argmin(errsum)], min(errsum)

def find_beam_waist(exp, run, masks, t, crop=[0,None,0,None], size=[5,5], pguess=[5, -0.3, 3, 0.5]):
    s0 = get_site_survival(exp, run, masks, t=t[0], crop=crop, size=size, output=False)

    focus = np.mean(s0[0].reshape(size[0],size[1]), axis=1)

    popt,pcov = curve_fit(gaussian, range(size[0]), focus, p0=pguess)
    x=np.linspace(0,size[0], 100)

    plt.figure()
    plt.plot(focus, 'ko')
    plt.plot(x, gaussian(x, *popt))
    plt.ylim(0,1)
    print('waist in atom spacing: ', 2*popt[2], '+-', 2*np.sqrt(np.diag(pcov))[2])
    return 2*popt[2], 2*np.sqrt(np.diag(pcov)[2])


# picture_num = 0,1 for getting counts in first or second image
def getAtomCounts(exp, run, masks, threshold, sortkey=0, crop=[0,None,0,None],  multiScan = False, img_idx=1, flag_idx=0, plot=True, surv_idx=2, flag_surv=False, fit=None, mode='emccd'):
    '''Calculates mean counts per atom.
    - img_idx: which image used to get the counts
    - flag_idx: which image used to flag the atoms'''

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    key = exp.key
    #determine whether there is an atom
    cut=10
    sig = exp.pics[:, crop[0]:crop[1], crop[2]:crop[3]]
    bkg = (np.mean(exp.pics[:, :cut, :-cut], axis=(1,2)) + np.mean(exp.pics[:, :cut, :-cut], axis=(1,2)) + np.mean(exp.pics[:, :cut, :-cut], axis=(1,2)) + np.mean(exp.pics[:, :cut, :-cut], axis=(1,2)))/4
    if mode == 'emccd':
        diff = sig
    else:
        diff = [np.subtract(s, bkg_arr[i%num_img]) for i,s in enumerate(sig)]
    roisums = np.array(list(map(lambda image:list(map(lambda mask:np.sum(mask*image),masks)),diff)))[img_idx::num_img]
    atomdata = get_binarized(exp, run, masks, threshold, crop=crop)[flag_idx::num_img] #flag atoms

    #get atom count data
    if flag_surv==True:
        survdata = get_binarized(exp, run, masks, threshold, crop=crop)[surv_idx::num_img]
        atomcounts = np.sum(np.multiply(survdata,atomdata)*roisums, axis=1)/np.sum(np.multiply(survdata,atomdata), axis=1)
    else:
        atomcounts = np.sum(atomdata*roisums, axis=1)/np.sum(atomdata, axis=1)

    cs_mean= [np.mean(atomcounts[i*exp.reps:(i+1)*exp.reps]) for i in range(len(exp.key))]
    cs_std= [np.std(atomcounts[i*exp.reps:(i+1)*exp.reps]) for i in range(len(exp.key))]

    #get background count data
    bkgcounts = np.sum((1-np.array(atomdata))*roisums, axis=1)/np.sum((1-np.array(atomdata)), axis=1)
    bkgcs_mean= [np.mean(bkgcounts[i*exp.reps:(i+1)*exp.reps]) for i in range(len(exp.key))]
    bkgcs_std= [np.std(bkgcounts[i*exp.reps:(i+1)*exp.reps]) for i in range(len(exp.key))]


    if multiScan == True:
        key = exp.key[:, sortkey]
        print(np.shape(key))
        if (isinstance(sortkey,int)):
            key_name = exp.key_name[sortkey]
        else:
            key_name = [exp.key_name[i] for i in sortkey]
    else:
        key = exp.key
        key_name = exp.key_name

    if (np.shape(key)[-1] == 2):

        #atom counts
        cs_mean_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), cs_mean))), key=lambda x: [x[0], x[1]]))
        cs_mean_sorted_reshape = np.reshape(cs_mean_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))
        cs_std_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), cs_std))), key=lambda x: [x[0], x[1]]))
        cs_std_sorted_reshape = np.reshape(cs_std_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))

        #bkg counts
        bkgcs_mean_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), bkgcs_mean))), key=lambda x: [x[0], x[1]]))
        bkgcs_mean_sorted_reshape = np.reshape(bkgcs_mean_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))
        bkgcs_std_sorted = np.array(sorted(np.transpose(np.vstack((np.transpose(key), bkgcs_std))), key=lambda x: [x[0], x[1]]))
        bkgcs_std_sorted_reshape = np.reshape(bkgcs_std_sorted[:,2], (len(np.unique(key[:, 0])), len(np.unique(key[:, 1]))))

        k0min = np.min(cs_mean_sorted[:,0])
        k0max = np.max(cs_mean_sorted[:,0])
        k1min = np.min(cs_mean_sorted[:,1])
        k1max = np.max(cs_mean_sorted[:,1])
        key0 = np.sort(np.unique(cs_mean_sorted[:,0]))
        key1 = np.sort(np.unique(cs_mean_sorted[:,1]))

        if (plot):
            fig, ax = plt.subplots(figsize=[5,4])
            im = plt.imshow(cs_mean_sorted_reshape, extent=[k1min, k1max, k0min, k0max],
                           aspect=(k1max - k1min)/(k0max - k0min), origin="lower")
            plt.xlabel(key_name[1])
            plt.ylabel(key_name[0])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('mean single atom counts')
            plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
            plt.show()

    else:

        key_sorted = np.sort(key)
        cs_mean_sorted = np.array(cs_mean)[np.argsort(exp.key)]
        cs_std_sorted = np.array(cs_std)[np.argsort(exp.key)]

        bkgcs_mean_sorted = np.array(bkgcs_mean)[np.argsort(exp.key)]
        bkgcs_std_sorted = np.array(bkgcs_std)[np.argsort(exp.key)]

        if (key.ndim == 1):
            fitFunc = None
            if fit == 'gaussian_peak':
                pguess = [np.max(cs_mean_sorted)-np.min(cs_mean_sorted),
                          key_sorted[np.argmax(cs_mean_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.min(cs_mean_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = gaussian
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[1], np.sqrt(np.diag(pcov))[1]))

            if fit == 'gaussian_dip':
                pguess = [-np.max(cs_mean_sorted)+np.min(cs_mean_sorted),
                          key_sorted[np.argmin(cs_mean_sorted)],
                          (key_sorted[-1]-key_sorted[0])/3,
                          np.max(cs_mean_sorted)]
                popt, pcov = curve_fit(gaussian, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = gaussian
                print('key fit = {:.5e} +/- {:.5e}'.format(popt[1], np.sqrt(np.diag(pcov))[1]))

            if fit == 'hockey_fall':
                pguess = [(key_sorted[-1]+key_sorted[0])/2,
                          (np.max(cs_mean_sorted)-np.min(cs_mean_sorted))/ (key_sorted[-1]-key_sorted[0]),
                          np.max(cs_mean_sorted)]
                popt, pcov = curve_fit(hockey_fall, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = hockey_fall
                print('key fit = {:.3e}'.format(popt[0]))

            if fit == 'hockey_rise':
                pguess = [(key_sorted[-1]+key_sorted[0])/2,
                          (np.max(cs_mean_sorted)-np.min(cs_mean_sorted))/ (key_sorted[-1]-key_sorted[0]),
                          np.max(cs_mean_sorted)]
                popt, pcov = curve_fit(hockey_rise, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = hockey_rise
                print('key fit = {:.3e}'.format(popt[0]))

            if fit == 'sinc2':
                # A*np.sinc(x-x0) + y0
                if (pguess == None):
                    pguess = [(key_sorted[-1]+key_sorted[0])/2, 0.8, 0, 0.01]
                popt, pcov = curve_fit(sinc2, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = sinc2
                print('x0, a0, y0, k0')
                print(popt)
                print('err ', np.sqrt(np.diag(pcov)))


            if fit == 'lor':
                # triplor(x, a0, a1, a2, kc, ks, x0, dx, y0)
                pguess = [0.4, 0.05,
                          (key_sorted[-1]+key_sorted[0])/2
                        ]
                popt, pcov = curve_fit(lor, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = lor
                print('a, k, x0')

                print(popt)

            if fit == 'decayt':
                # decayt(t, tau, a, y0): y0 + a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(cs_mean_sorted)-np.min(cs_mean_sorted), np.min(cs_mean_sorted)]
                popt, pcov = curve_fit(decayt, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = decayt
                print('tau, a, y0 ', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt_gaussian':
                # decayt(t, tau, a, y0): y0 + a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(cs_mean_sorted)-np.min(cs_mean_sorted), np.min(cs_mean_sorted)]
                popt, pcov = curve_fit(decayt_gaussian, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = decayt_gaussian
                print('tau, a, y0 ', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt0':
                # decayt0(t, tau, a): a*np.exp(-t/tau)
                pguess = [key_sorted[-1]/5, np.max(cs_mean_sorted)-np.min(cs_mean_sorted)]
                popt, pcov = curve_fit(decayt0, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = decayt0
                print('t, tau, a', popt)
                print('err ', np.sqrt(np.diag(pcov)))

            if fit == 'decayt1':
                # decayt1(t, tau, a): a*np.exp(-t/tau)+1
                pguess = [key_sorted[-1]/5, np.max(cs_mean_sorted)-np.min(cs_mean_sorted)]
                popt, pcov = curve_fit(decayt1, key_sorted, cs_mean_sorted, p0=pguess)
                fitFunc = decayt1
                print('t, tau, a', popt)
                print('err ', np.sqrt(np.diag(pcov)))


            if (plot):
                key_fine = np.linspace(key_sorted[0], key_sorted[-1], 200, endpoint=True)
                fig, ax = plt.subplots(figsize=[5,4])
                if (fitFunc != None):
                    plt.plot(key_fine, fitFunc(key_fine, *popt), 'k-')
                plt.plot(key_sorted, cs_mean_sorted, color='k', marker='o', linestyle=':', alpha=0.7)
                plt.xlabel(key_name)
                plt.ylabel('mean single atom counts')
                plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
                plt.show()

            if (fitFunc != None):
                Nexp = fitFunc(key_sorted, *popt)
                r = cs_mean_sorted - Nexp
                RSS = np.divide(r,cs_mean_sorted) ** 2
                chisq = np.sum(RSS)
                redchisq = chisq/(len(cs_mean_sorted)-len(pguess)+1)
                print('red chisq', redchisq)

    if (np.shape(key)[-1] == 2):

        return cs_mean_sorted_reshape, cs_std_sorted_reshape, bkgcs_mean_sorted_reshape, bkgcs_std_sorted_reshape

    else:
        return cs_mean_sorted, cs_std_sorted, bkgcs_mean_sorted, bkgcs_std_sorted


def newDay():
    """Creates a new notebook (and folder if missing) for current day, from the file that this funciton is called from."""
    src_dir = os.getcwd()
    src_filename = src_dir.replace('B:\\Yb heap\\Yb_data\\','')+'.ipynb'

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


# picture_num = 0,1 for getting counts in first or second image
def getImageLossError(exp, run, masks, threshold, sortkey=0, crop=[0,None,0,None],  multiScan = False, flag_idx=0, img_idx=1, surv_idx=2, flag_surv=False):
    '''Calculates mean counts per atom.
    - img_idx: which image used to get the counts
    - flag_idx: which image used to flag the atoms'''

    cs_mean, cs_std, bkgcs_mean, bkgcs_std = getAtomCounts(exp, run, masks, threshold, sortkey=sortkey, crop=crop,  multiScan = multiScan, img_idx=img_idx, flag_idx=flag_idx, plot=False, flag_surv=flag_surv)
    s = var_scan_survprob(exp, run, masks, threshold, fit='n', sortkey=sortkey,crop=crop,  multiScan = multiScan, keep_img=[flag_idx, surv_idx], plot=False)

    #find infidelity from the counts measurement (overlap of two gaussians)
    def gausOverlap(a, b, c, d):
        return NormalDist(mu=a, sigma=b).overlap(NormalDist(mu=c, sigma=d))

    gausOverlap = np.vectorize(gausOverlap)
    inf = gausOverlap(cs_mean, cs_std, bkgcs_mean, bkgcs_std)

    #find loss from survival measurement, and correct for infidelity
    loss = 1-np.array(s[0])-inf

    #error loss quadrature
    error = np.sqrt(inf**2+loss**2)

    if multiScan == True:
        key = exp.key[:, sortkey]
        print(np.shape(key))
        if (isinstance(sortkey,int)):
            key_name = exp.key_name[sortkey]
        else:
            key_name = [exp.key_name[i] for i in sortkey]
    else:
        key = exp.key
        key_name = exp.key_name

    if (np.shape(key)[-1] == 2):

        k0min = np.min(exp.key[:,0])
        k0max = np.max(exp.key[:,0])
        k1min = np.min(exp.key[:,1])
        k1max = np.max(exp.key[:,1])
        key0 = np.sort(np.unique(exp.key[:,0]))
        key1 = np.sort(np.unique(exp.key[:,1]))

        fig, ax = plt.subplots(figsize=[5,4])
        im = plt.imshow(error, extent=[k1min, k1max, k0min, k0max],
                       aspect=(k1max - k1min)/(k0max - k0min), origin="lower")
        plt.xlabel(key_name[1])
        plt.ylabel(key_name[0])
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('quadrature error of loss and infidelity')
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.show()

    else:

        key_sorted = np.sort(key)
        plt.plot(key_sorted, error)
        if (type(exp.key_name) == str):
            plt.xlabel(exp.key_name)
        else:
            plt.xlabel(exp.key_name[sortkey])
        plt.ylabel('mean single atom counts')
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")

    return error

def cameraCountsToPhotons(counts, l=556, useOffset=True):
    qe = exc.QE556
    if (l==399):
        qe = exc.QE399
    if (useOffset == True):
        offset = exc.offset
    else:
        offset = 0

    return (counts - offset)*exc.CMOSsensitivity12bit/qe

def getMasksAlign(img, disttozero=[-50, -50, 30], t=0.3, output=True, N=[16,3], wmask=2.5, fftN=2500, mindist=30):

    fimg = np.fft.fft2(img, s = (fftN,fftN))
    fimg = np.fft.fftshift(fimg)
    fimgAbs = np.abs(fimg)
    fimgArg = np.angle(fimg)

    # fimgMax = ndi.maximum_filter(fimg, size = 100, mode = 'constant')
    fMaxCoord = peak_local_max(fimgAbs, min_distance=mindist, threshold_rel=t)
    if (output):
        print(len(fMaxCoord))
    # fMaxBool = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1, num_peaks = 4, indices=False)

    fMaxCoord = fMaxCoord[fMaxCoord[:,0]-fftN/2>disttozero[0]]# Restrict to positive quadrant
    fMaxCoord = fMaxCoord[fMaxCoord[:,1]-fftN/2>disttozero[1]] # Restrict to positive quadrant
    fMaxCoord = fMaxCoord[fMaxCoord.sum(axis=1)-fftN>disttozero[2]] # Restrict to positive quadrant

    xsort = np.argsort(fMaxCoord[:,1]+fMaxCoord[:,0]/5)
    ysort = np.argsort(fMaxCoord[:,0]+fMaxCoord[:,1]/5)

    xpeak, ypeak = fMaxCoord[xsort[0]], fMaxCoord[ysort[0]]

    if (output):
        print(xpeak, ypeak)

        fig, ax = plt.subplots()
        plt.imshow(fimgAbs)
        plt.plot([xpeak[1]],[xpeak[0]],'r.')
        plt.plot([ypeak[1]],[ypeak[0]],'b.')
        # plt.imshow(fimgAbs[700:1000, 700:1000])
        plt.vlines(fftN/2+disttozero[0],0, fftN, colors='b', linestyles='-')
        plt.hlines(fftN/2+disttozero[1], 0, fftN, colors='r', linestyles='-')

    freqs = np.fft.fftfreq(fftN)
    freqs = np.fft.fftshift(freqs)
    fx, fy = freqs[xpeak], freqs[ypeak]
    dx, dy = 1/fx, 1/fy

    phix, phiy = fimgArg[xpeak[0], xpeak[1]], fimgArg[ypeak[0], ypeak[1]]

    normX = np.sqrt(np.sum(fx**2))
    dx = (1/normX)*(fx/normX)

    normY = np.sqrt(np.sum(fy**2))
    dy = (1/normY)*(fy/normY)

    dx[1]=-dx[1]
    dy[0]=-dy[0]
    tmp = dy[0]
    dy[0] = dx[1]
    dx[1] = tmp

    nsx = np.arange(N[1])
    nsy = np.arange(N[0])

    px = arr([(dx*ind) for ind in nsx])
    py = arr([(dy*ind) for ind in nsy])

    pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N[0]*N[1] ,2))
    x = np.arange(len(img[0]))
    y = np.arange(len(img[:,0]))

    xx, yy = np.meshgrid(x, y)
    masks = arr([psf(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])

    if (output):
        fig, ax = plt.subplots(figsize=[20,8])
        plt.imshow(img)
        plt.plot(pts[:,1],pts[:,0],'r.', markersize=2)

    return dx, dy, [(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)], pts, masks


def getFitPts(img, pts, c):
    ptsfit = []
    for pt in pts:
    #         print(pt)
        x1, x2, y1, y2 = int(pt[1]-c), int(pt[1]+c+1), int(pt[0]-c), int(pt[0]+c+1)
        impt = img[y1:y2,x1:x2]
    #         plt.imshow(impt)
        ptfit = gaussFit2d_rot(impt)
        ptsfit.append([int(pt[1]-c)+ptfit[1][1], int(pt[0]-c)+ptfit[1][2]])
    return np.array(ptsfit)

def getMeanSpacingsAngles(ptsfit, rows=3, cols=16):
    dxmx, dxmy, dymx, dymy = 0, 0, 0, 0

    for j in range(rows):
        dxmx += ptsfit[j*cols+1:(j+1)*cols,0] - ptsfit[j*cols:(j+1)*cols-1,0]
        dxmy += ptsfit[j*cols+1:(j+1)*cols,1] - ptsfit[j*cols:(j+1)*cols-1,1]
    dxmx = np.mean(dxmx/rows)
    dxmy = np.mean(dxmy/rows)

    for j in range(rows-1):
        dymx += ptsfit[(j+1)*cols:(j+2)*cols,0]- ptsfit[j*cols:(j+1)*cols,0]
        dymy += ptsfit[(j+1)*cols:(j+2)*cols,1] - ptsfit[j*cols:(j+1)*cols,1]
    dymx = np.mean(dymx/(rows-1))
    dymy = np.mean(dymy/(rows-1))
    return [dxmx, dxmy, dymx, dymy]

def getArrayCenter(ptsfit, rows=3, cols=16):
    return np.mean(ptsfit[:,0]), np.mean(ptsfit[:,1])

def getMasksForChimeraLoadOnly(pts, s, csize=5):
    wmask=2.5
    x = np.arange(s[0])
    y = np.arange(s[1])

    yy, xx = np.meshgrid(y, x)
    masks = arr([psf(np.sqrt((yy-pts[i,0])**2+(xx-pts[i,1])**2), wmask) for i in range(len(pts))])
    cropPts = arr([[int(round(pt[1])-csize), int(round(pt[1])+csize), int(round(pt[0])-csize), int(round(pt[0])+csize)] for pt in pts])
    masksCropped = arr([mask[c[0]:c[1],c[2]:c[3]] for c, mask in zip(cropPts, masks)])
    return masksCropped, cropPts, masks

def getMasks(mimg, pts, mod2Dgauss=[2.0,2.0], wmask = 3, wmask2=1, mode = 'gauss',  output = True, coords = None,
                     get_mask_centers = False):
    """Given an averaged atom image, returns list of masks, where each mask corresponds to the appropriate mask for a single atom."""

    chor = mod2Dgauss[0]
    cver = mod2Dgauss[1]
    ptsfit = []
    i = 0
    # fig, ax = plt.subplots(figsize=[30,30])
    for pt in pts:
#        print(pt)
        x1, x2, y1, y2 = int(pt[1]-chor), int(pt[1]+chor+1), int(pt[0]-cver), int(pt[0]+cver+1)
        impt = mimg[y1:y2,x1:x2]

        # ax = plt.subplot(10,2,i)
        # plt.imshow(impt)
        # plt.show()
        #

        ptfit = gaussFit2d_rot(impt)

        if np.sqrt(np.diag(ptfit[2]))[0] <100:
            ptsfit.append([int(pt[0]-cver)+ptfit[1][2],int(pt[1]-chor)+ptfit[1][1]])
        else:
            ptsfit.append(pt)
    pts = np.array(ptsfit)
    plt.show()

    if output == True:
        fig, ax = plt.subplots(figsize=[30,30])
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
    if mode == 'gaussEllipse':
        masks = arr([psf_ellipse((xx-pts[i,1]),(yy-pts[i,0]), wmask2, wmask) for i in range(len(pts))])
        if coords != None:
            masks = arr([psf(np.sqrt((xx-coords[1])**2+(yy-coords[0])**2), wmask) for i in range(len(pts))])
    if mode == 'box':
        masks = arr([box(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
    if output == True:
        fig, ax = plt.subplots(figsize=[30,30])
        plt.imshow(np.sum(masks, axis=0))
        if coords != None:
            plt.plot(coords[1],coords[0],'r.')
        else:
            plt.plot(pts[:,1],pts[:,0],'r.')

        plt.show()

    # plt.rcParams["figure.figsize"] = (5,4)

    if (get_mask_centers == True):
        return masks, pts
    else:
        return masks
