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

def fit_histogram(x, y, threshold_guess, pguess=[]):
    idx_guess = np.abs(x - threshold_guess).argmin()
    if pguess:
        p0=pguess
    else:
        p0 = np.array([ 1.1, 4900,  0.015, np.max(y[:idx_guess]),  
                           np.max(y[idx_guess:]), 5900, 400, 1.8])
    
    popt, pcov = curve_fit(emccd_hist_skew, x, y, p0=p0)

    mistake_arr = []
    idx_voidpeak = np.argmax(y[:idx_guess])
    idx_atompeak = np.argmax(y[idx_guess:])
    threshold_arr = arr(np.arange(x[idx_voidpeak], x[idx_atompeak]), dtype=int)

    xfit = arr(np.arange(threshold_arr[0]-200, threshold_arr[-1]+1500), dtype=int)
    atom_curve = gaussian_skew(xfit, popt[4], popt[5], popt[6], popt[7], 0)
    void_curve = emccd_bkg(xfit, popt[0], popt[1], popt[2], popt[3])

    for t in threshold_arr:
        tidx = t - xfit[0]
        av_mistake = np.sum(atom_curve[:tidx+1])
        va_mistake = np.sum(void_curve[tidx:])
        mistake_arr.append([va_mistake, av_mistake])

    tot_area_a = np.sum(atom_curve)
    tot_area_v = np.sum(void_curve)

    threshold_idx = np.argmin(np.sum(mistake_arr, axis=1))
    threshold = threshold_arr[threshold_idx]

    va_mistake, av_mistake = mistake_arr[threshold_idx]
    
    return threshold, [va_mistake/tot_area_v, av_mistake/tot_area_a], popt, pcov, xfit


def analyze_histogram(exp, run, masks, threshold_guess = 5300, bin_width = 10, keep_img = [0,1], crop = [0,None,0,None], single_threshold=True, pguess=None):

    hist_xdata_all = []
    hist_all = []
    threshold_all = []
    popt_all = []
    pcov_all = []
    inf_all = []
    xfit_all = []

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    # analyze each image seperately
    for num in keep_img:

        #find counts from an image
        sig = exp.pics[num::num_img, crop[0]:crop[1], crop[2]:crop[3]]
        cs = np.array(list(map(lambda image: list(map(lambda mask:np.sum(mask*image),masks)),sig))).flatten()

        # sort counts into histogram
        hist, bin_edges = np.histogram(cs, bins=np.arange(np.min(cs)-bin_width/2, np.max(cs)+bin_width/2, bin_width))
        hist_xdata = bin_edges[:-1]

        hist_xdata_all.append(hist_xdata)
        hist_all.append(hist)

        if num == 0:
            threshold, inf, popt, pcov, xfit = fit_histogram(hist_xdata, hist, threshold_guess, pguess=pguess)
            if single_threshold==True:
                threshold0 = threshold
                popt0 = popt
                pcov0 = pcov
                inf0 = inf
                xfit0 = xfit
                
        else:
            if single_threshold==True:
                threshold = threshold0
                popt = popt0
                pcov = pcov0
                inf = inf0
                xfit = xfit0
            else:
                threshold, inf, popt, pcov, xfit = fit_histogram(hist_xdata, hist, threshold_guess, pguess=pguess)
                
        inf_all.append(inf)
        popt_all.append(popt)
        pcov_all.append(pcov)
        threshold_all.append(threshold)
        xfit_all.append(xfit)

    fig, ax = plt.subplots(figsize=[6,4])
    alphaarr = np.linspace(1.0,0.5, num_img)
    Ncl = 5
    c_ls = plt.get_cmap('rainbow', Ncl)  
    for num, n in enumerate(keep_img):
        ax.plot(hist_xdata_all[num], hist_all[num], alpha=alphaarr[num], drawstyle='steps-post', color='k', zorder = 0)
        ax.plot([threshold_all[num], threshold_all[num]], [np.min(hist_all[num]), np.max(hist_all[num])], linestyle='--', color=c_ls(n/Ncl), label=n, zorder = 0)
        ax.plot(hist_xdata_all[num], emccd_hist_skew(hist_xdata_all[num], *popt_all[num]), linewidth=2, color=c_ls(n/Ncl))

    ax.set_xlabel('Counts collected')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    ax.set_ylim(0.7, )
    plt.legend()
    ax.set_title(str(exp.data_addr) + "data_" + str(run) + ".h5")
    plt.show()

    format_string = "{:.3f}"

    print('bkg peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,1]])
    print('atom peak position: ',[format_string.format(number) for number in np.array(popt_all)[:,5]])
    print('atom peak width: ',[format_string.format(number) for number in np.array(popt_all)[:,6]])
    print('bkg peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,3]])
    print('atom peak amplitude: ',[format_string.format(number) for number in np.array(popt_all)[:,4]])
    print('thresholds: ',[format_string.format(number) for number in [t for i,t in enumerate(threshold_all)]])
    print('va fitted infidelity: ',[format_string.format(number*100) for i,number in enumerate(arr(inf_all)[:,0])], 'percent')
    print('av fitted infidelity: ',[format_string.format(number*100) for i,number in enumerate(arr(inf_all)[:,1])], 'percent')

    return threshold_all, hist_xdata_all, hist_all, popt_all, pcov_all, inf_all, xfit_all

def get_events(exp, run, masks, threshold, crop=[0,None,0,None], keep_img=[0,1], skipFirst=True):

    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key)) #number of images per rep
    data = exp.pics[:, crop[0]:crop[1], crop[2]:crop[3]] #get data

    if skipFirst==True:
        key = exp.key[1:]
        data = data[exp.reps*num_img:]
        key_name = exp.key_name
    else:
        data = data
        key = exp.key
        key_name = exp.key_name

    roisums = np.array(list(map(lambda image:list(map(lambda mask:np.sum(mask*image),masks)),data))) #sum counts from atom positions

    atoms = []
    for idx, img in enumerate(keep_img): #for each mask/rep find whether there is an atom or not
        atoms.append([np.clip(r, threshold[idx], threshold[idx]+1) - threshold[idx] for i,r in enumerate(roisums[img::num_img])])
    atoms = arr(atoms)

    if len(keep_img)==2: #for now consider only 2 image case
        sums = atoms[0]+atoms[1]
        diffs = atoms[0]-atoms[1]
        aa = np.greater(sums, 1).astype(int) #if atom on both images sum=2
        vv = np.less(sums, 1).astype(int) #if atom on neither image sum=0
        av = np.equal(diffs, 1).astype(int) #if atom on first image only diff=+1
        va = np.less(diffs, 0).astype(int) #if atom on second image only diff=-1

        aaf, vvf, avf, vaf = np.sum(aa), np.sum(vv), np.sum(av), np.sum(va) #sum over ROIs and repetitions.
        

    return aa, av, va, vv


def spam_matrix(pva1, pav1, pva2, pav2, piloss, pvloss, pba, ppump):
   
    M = [[0]*4 for i in range(4)]
    M[0][0] = 1-pav1-pav2-2*piloss-pvloss-pba-ppump
    M[0][1] = pva2+ppump
    M[0][2] = 0
    M[0][3] = 0
    
    M[1][0] = pav2+piloss+pvloss+pba+ppump
    M[1][1] = 1-pav1-pva2-piloss-ppump
    M[1][2] = 0
    M[1][3] = pva1
       
    M[2][0] = pav1
    M[2][1] = 0
    M[2][2] = 1
    M[2][3] = pva2
     
    M[3][0] = 0
    M[3][1] = pav1+piloss
    M[3][2] = 0
    M[3][3] = 1-pva1-pva2
    
    return M, np.linalg.inv(M)

def extract_invMerror(inv_M, dist, piloss, piloss_err, pvloss, pvloss_err, pba, pba_err, ppump, ppump_err):

    Merr = [[0]*4 for i in range(4)]
    
    pva1_dist = arr(dist)[0,:, 0]
    pav1_dist = arr(dist)[0,:, 1]
    pva2_dist = arr(dist)[1,:, 0]
    pav2_dist = arr(dist)[1,:, 1]

    Ntot = len(pva1_dist)

    piloss_dist = np.random.normal(loc=piloss, scale=piloss_err, size=Ntot).transpose()
    pvloss_dist = np.random.normal(loc=pvloss, scale=pvloss_err, size=Ntot).transpose()
    pba_dist = np.random.normal(loc=pba, scale=pba_err, size=Ntot).transpose()
    ppump_dist = np.random.normal(loc=ppump, scale=ppump_err, size=Ntot).transpose()

    inv_Marr = []
    for N in range(Ntot):

        pva1 = pva1_dist[N]
        pav1 = pav1_dist[N]
        pva2 = pva2_dist[N]
        pav2 = pav2_dist[N]
        piloss = piloss_dist[N]
        pvloss = pvloss_dist[N]
        pba = pba_dist[N]
        ppump = ppump_dist[N]

        M_N, inv_M_N = spam_matrix(pva1, pav1, pva2, pav2, piloss, pvloss, pba, ppump)

        inv_Marr.append(inv_M_N)

    inv_Marr = arr(inv_Marr)

    fig, ax = plt.subplots(4,4, figsize=(8,8))
    for i in range(4):
        for j in range(4):
            y = inv_Marr[:,i,j]
            bin_width = (np.max(y)-np.min(y))/100
            if bin_width==0:
                bins_ls = [np.max(y)]
            else:
                bins_ls = np.arange(np.min(y)-bin_width/2, np.max(y)+bin_width/2, bin_width)
            f, bin_edges = np.histogram(y, bins=bins_ls)
            x = bin_edges[:-1]      
            ax[i,j].plot(x, f, drawstyle='steps-post')
            err = np.std(y)
            ax[i,j].axvline(inv_M[i,j]+err)
            ax[i,j].axvline(inv_M[i,j]-err)
            ax[i,j].axvline(inv_M[i,j], linestyle=':')
            
            Merr[i][j] = err

    plt.tight_layout()
    
    return Merr

def extract_inferror(t, keep_img = [0,2], n_ls = 10000):
    
    output = []
    dist = []
    
    for j,k in enumerate(keep_img):
        
        errorbars = []
    
        threshold = t[0][j]
        popt = t[3][j]
        pcov = t[4][j]
        xfit = t[6][j]
        inf_all = t[5][j]
 
        z = np.random.multivariate_normal(mean=popt.reshape(len(popt),), cov=pcov, size=n_ls)
        y = np.transpose(z)
        inf_err = []

        for n in range(n_ls):
            popt = y[:,n]
            fit = emccd_hist_skew(xfit, *popt)

            atom_curve = gaussian_skew(xfit, popt[4], popt[5], popt[6], popt[7], 0)
            void_curve = emccd_bkg(xfit, popt[0], popt[1], popt[2], popt[3])

            tidx=threshold- xfit[0]
            av_mistake = np.sum(atom_curve[:tidx+1])
            va_mistake = np.sum(void_curve[tidx:])

            tot_area_a = np.sum(atom_curve)
            tot_area_v = np.sum(void_curve)

            inf_err.append([va_mistake/tot_area_v, av_mistake/tot_area_a])
            
        fig, ax = plt.subplots(1,2, sharey=True, figsize=(5, 2))
        ax = ax.flatten()
        for n in range(2):

            y = arr(inf_err)[:,n]
            bin_width = (np.max(y)-np.min(y))/100
            if bin_width==0:
                bins_ls = [np.max(y)]
            else:
                bins_ls = np.arange(np.min(y)-bin_width/2, np.max(y)+bin_width/2, bin_width)
            va_y, bin_edges = np.histogram(y, bins=bins_ls)
            va_x = bin_edges[:-1]                          

            mean = inf_all[n]

            ratio_plus = []
            ratio_minus = []

            mean_idx = abs(va_x-mean).argmin()
            for i, v in enumerate(va_x):

                if i<mean_idx:
                    sum1dminus = np.sum(va_y[i:mean_idx]) 
                    sum1dminustot = np.sum(va_y[:mean_idx]) 
                    ratio_minus.append(sum1dminus/sum1dminustot)

                if i>mean_idx:
                    sum1dplus = np.sum(va_y[mean_idx:i]) 
                    sum1dplustot = np.sum(va_y[mean_idx:]) 
                    ratio_plus.append(sum1dplus/sum1dplustot)

            errors = [abs(mean-va_x[abs(arr(ratio_minus)-0.95).argmin()]), 
                      abs(mean-va_x[abs(arr(ratio_plus)-0.95).argmin()+mean_idx])]
            
            errorbars.append(errors)

            ax[n].plot(va_x, va_y, drawstyle='steps-post')
            ax[n].axvline(mean-errors[0])
            ax[n].axvline(mean+errors[1])
            ax[n].axvline(mean, linestyle=':')
            ax[n].set_xlim(0, max(va_x))
            ax[n].set_ylim(0,)
        ax[0].set_title('pva, img %i' %k)
        ax[1].set_title('pav, img %i' %k)
        
        output.append([inf_all, errorbars])
        dist.append(inf_err)
        
    return output, dist


def SPAM_correct(address, run, masks, crop, threshold, inv_M, keep_img, skipFirst):    
    '''
    dataAddress, run, masks for the experiment that is meant to be analyzed
    thrsehold and inv_M should have been determined from SPAM measurement
    '''
    
    exp = ExpFile(address / 'Raw Data', run)
    num_img = int(exp.pics.shape[0]/exp.reps/len(exp.key))

    aa, av, va, vv = get_events(exp, run, masks, threshold=threshold, crop=crop,
                    keep_img=keep_img, skipFirst = skipFirst)

    if skipFirst==True:
        key = exp.key[1:]
    else:
        key = exp.key

    aaff = np.sum(np.sum(aa.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    avff = np.sum(np.sum(av.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    vaff = np.sum(np.sum(va.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    vvff = np.sum(np.sum(vv.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)

    real_arr = []
    measured_arr = []
    for i in range(len(key)):

        measured = arr([aaff[i], avff[i], vaff[i], vvff[i]])
        real = np.matmul(inv_M, measured)

        real_arr.append(real)
        measured_arr.append(measured)
    real_arr = arr(real_arr)
    
    return real_arr, measured_arr


def analyze_survival(exp, run, masks, threshold, crop, fit, pguess, keep_img=[0,1], skipFirst=True, fullscale=True, givemean=True):

    aa, av, va, vv = get_events(exp, run, masks, threshold, crop=crop,
                    keep_img=keep_img, skipFirst = skipFirst)
    
    if skipFirst==True:
        key = exp.key[1:]
    else:
        key = exp.key
    
    aaff = np.sum(np.sum(aa.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    avff = np.sum(np.sum(av.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    vaff = np.sum(np.sum(va.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    vvff = np.sum(np.sum(vv.reshape(len(key), exp.reps, len(masks)), axis=1), axis=1)
    
    s = aaff/(aaff+avff)
    serr = np.sqrt(s*(1-s)/(aaff+avff))
        
    if givemean:
        aaff_sum = np.sum(aaff)
        avff_sum = np.sum(avff)
        s_sum = aaff_sum/(aaff_sum+avff_sum)
        serr_sum = np.sqrt(s_sum*(1-s_sum)/(aaff_sum+avff_sum))

        print("mean survival: %.3f +- %.3f percent" %(s_sum*1e2, serr_sum*1e2))

    if fit:
        gmodel = Model(fit)
        fit_params = Parameters()
        for i,p in enumerate(pguess):
            fit_params.add(p[0], value=p[1])

        result = gmodel.fit(s, fit_params, t=key, weights=1/serr**2, nan_policy='omit')
        param = []
        param_err = []
        for k in result.params:
            param.append(result.params[k].value)
            param_err.append(result.params[k].stderr)
        x_fine = np.linspace(min(key), max(key),100000)
        fit_fine = fit(x_fine, *param)
        
        print(arr(pguess)[:,0], '=', param)
        print('Err', arr(pguess)[:,0], '=', param_err)

        plt.figure()
        plt.title(str(exp.data_addr) + "data_" + str(run) + ".h5")
        plt.plot(x_fine, fit_fine, 'k-')
        plt.errorbar(key, s, serr, marker='o', linestyle='none', color='k', mfc='white')
        if fullscale==True:
            plt.ylim(0, 1)
        plt.ylabel('survival')
        plt.xlabel(exp.key_name)
        plt.show()

    if givemean and fit:     
        return key, s, serr, param, param_err, s_sum, serr_sum
    elif fit:
        return key, s, serr, param, param_err
    elif givemean:     
        return key, s, serr, s_sum, serr_sum