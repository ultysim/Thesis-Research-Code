from sklearn.model_selection import train_test_split
import lmfit
from lmfit import minimize, Parameters, fit_report
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import Analysis
from scipy.optimize import least_squares
import sys


def sigmoid(x,v0):
    return 1.0 / (1 + np.exp(np.array((-x+v0)/0.1,dtype='float')))

def model(params,x,v0):

    model_l = params[0] + params[1] * np.sin(x * params[2] + (params[3] + params[4] * np.pi)) + params[5] * (x - v0)
    model_r = params[0] + params[1] * np.sin(x * params[2] + params[3]) + params[5] * (x - v0)

    return model_l,model_r

def residual(params, x, y, v0, eps_data):

    model_l,model_r = model(params,x,v0)
    
    out_r = (y-model_r)*sigmoid(x,v0)
    out_l = (y-model_l)*(1.0-sigmoid(x,v0))
    
    return (out_r+out_l)/eps_data

def residual_new(params, x, y, v0, eps_data):
#Conditional residual function to catch multislip events
    model_l,model_r = model(params,x,v0)

    out_r = (y-model_r)
    out_l = (y-model_l)


    for i in range(len(out_l)/2):
        if abs(model_r[i] - model_l[i]) > 0.08:
            if abs(y[i]-model_r[i]) < 0.04:
                out_l[i] = y[i]-model_r[i]

    for i in range(len(out_r)/2,len(out_r)):
        if abs(model_r[i] - model_l[i]) > 0.08:
            if abs(y[i]-model_l[i]) < 0.04:
                out_r[i] = y[i]-model_l[i]

    out_r *= sigmoid(x,v0)
    out_l *= (1.0-sigmoid(x,v0))

    
    return (out_r+out_l)/eps_data

def search(seq, t):  
    min = 0
    max = len(seq) - 1
    while True:        
        m = (min + max) // 2
        if max <= min:
            if np.abs(t-seq[m]) < np.abs(t-seq[m-1]):          
                return m
            else:
                return m-1        
        if seq[m] < t:
            min = m + 1
        elif seq[m] > t:
            max = m - 1
        else:
            return m


def plot(data,result,full=False):
    x_min = result['x_min']
    x_max = result['x_max']
    if full:
        x = data.x
        y = data.y
    else:
        i_min = search(data.x,x_min)
        i_max = search(data.x,x_max)+1

        x = data.x[i_min:i_max]
        y = data.y[i_min:i_max]

    v0 = (x_min+x_max)/2.0

    params = result['result'].x

    model_l,model_r = model(params,x,v0)

    plt.plot(x,y)
    plt.plot(x,model_r,c='r')
    plt.plot(x,model_l,c='k')

def multi_reg_param_ci(data, x_min, x_max, result, eps): 
    i_min = search(data.x,x_min)
    i_max = search(data.x,x_max)+1

    if i_min < 0:
        i_min = 0
    if i_max > len(data.x):
        i_max = len(data.x)

    N = float(i_max - i_min)
    k = len(result.x)

    x = data.x[i_min:i_max]
    y = data.y[i_min:i_max]

    v0 = (x_min+x_max)/2.0

    res = result.fun*eps #check on this!!

    sigma_r = 1/(N-k)*np.sum(res**2)
    
    inv_j = np.linalg.inv(np.dot(result.jac.transpose(),result.jac))

    cov = sigma_r*inv_j
    
    return np.sqrt(np.diag(cov))

def regression_CV(x_min, x_max, data, residual = residual, loss = 'linear', eps = 0.1, val_split = 0.2, plot=False, x_scale = 1.0, y_tol=0.04):
    
    x = data.x
    y = data.y
    
    i_min = search(x,x_min)
    i_max = search(x,x_max)+1
    
    if i_min < 0:
        i_min = 0
    if i_max > len(x):
        i_max = len(x)
    
    x = x[i_min:i_max]
    y = y[i_min:i_max]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = val_split)
    
    
    a = (np.percentile(y_train,99)-np.percentile(y_train,1))/2.0
    
    params = []
    bounds = ([],[])
    
    h = np.mean(y_train)+np.random.randn()*0.1
    params.append(h) #height
    bounds[0].append(h*0.7)
    bounds[1].append(h*1.3)
    
    params.append(a+np.random.randn()*0.001) #amplitude
    bounds[0].append(a*0.85)
    bounds[1].append(a*1.15)
    
    params.append(2*np.pi/9+np.random.randn()*0.01) #frequency
    bounds[0].append(0.6)
    bounds[1].append(0.8)
    
    params.append(np.random.random()+0.001) #phase
    bounds[0].append(0.0)
    bounds[1].append(2*np.pi)
    
    params.append(np.random.uniform(-1.9,1.9)) #delta phase
    bounds[0].append(-2.0)
    bounds[1].append(2.0)
    
    params.append(np.random.randn()*0.0001) #slope
    bounds[0].append(-0.001)
    bounds[1].append(0.001)
    
    v0 = (x_min+x_max)/2.0
    
    try:
    
        out = least_squares(residual, x0 = params, bounds=bounds, loss=loss, x_scale = x_scale, args=(X_train , y_train, v0, eps))
    
        val_acc = validation(out,v0,X_test,y_test,y_tol)
        
        if val_acc is None:
            out = None
        elif not 0.155 < out['x'][4] < 1.845:
            out = None
            val_acc = None
        
    except TypeError as e:
        print(e)
        out, val_acc = None, None

    except:
        print(sys.exc_info()[0])
        out, val_acc = None, None
    
    if plot:
        params = out['x']
        model_l,model_r = model(params,x,v0)
        plt.plot(x,y)
        plt.plot(x,model_r,c='r')
        plt.plot(x,model_l,c='k')
    
    return out, val_acc    


def regression_CV_loss(x_min, x_max, data, residual = residual, loss = 'linear', eps = 0.1, val_split = 0.2, plot=False, x_scale = 1.0):
    
    x = data.x
    y = data.y
    
    i_min = search(x,x_min)
    i_max = search(x,x_max)+1
    
    if i_min < 0:
        i_min = 0
    if i_max > len(x):
        i_max = len(x)
    
    x = x[i_min:i_max]
    y = y[i_min:i_max]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = val_split)
    
    
    a = (np.percentile(y_train,99)-np.percentile(y_train,1))/2.0
    
    params = []
    bounds = ([],[])
    
    h = np.mean(y_train)+np.random.randn()*0.1
    params.append(h) #height
    bounds[0].append(h*0.7)
    bounds[1].append(h*1.3)
    
    params.append(a+np.random.randn()*0.001) #amplitude
    bounds[0].append(a*0.85)
    bounds[1].append(a*1.15)
    
    params.append(2*np.pi/9+np.random.randn()*0.01) #frequency
    bounds[0].append(0.6)
    bounds[1].append(0.8)
    
    params.append(np.random.random()+0.001) #phase
    bounds[0].append(0.0)
    bounds[1].append(2*np.pi)
    
    params.append(np.random.uniform(-1.9,1.9)) #delta phase
    bounds[0].append(-2.0)
    bounds[1].append(2.0)
    
    params.append(np.random.randn()*0.0001) #slope
    bounds[0].append(-0.001)
    bounds[1].append(0.001)
    
    v0 = (x_min+x_max)/2.0
    
    try:
    
        out = least_squares(residual, x0 = params, bounds=bounds, loss=loss, x_scale = x_scale, args=(X_train , y_train, v0, eps))
    	
        #Take the inverse of the loss, still try to maximize
        val_acc = 1.0/validation_loss(out,v0,X_test,y_test)

        
        if val_acc is None:
            out = None
        elif not 0.155 < out['x'][4] < 1.845:
            out = None
            val_acc = None
        
    except ZeroDivisionError:
        out, val_acc = None, None

    except TypeError as e:
        print(e)
        out, val_acc = None, None

    except:
        print(sys.exc_info()[0])
        out, val_acc = None, None
    
    if plot:
        params = out['x']
        model_l,model_r = model(params,x,v0)
        plt.plot(x,y)
        plt.plot(x,model_r,c='r')
        plt.plot(x,model_l,c='k')
    
    return out, val_acc


def validation(out,v0,x,y,y_tol=0.04):
    params = out['x']
    count = list()
    for i in range(len(x)):
        model_l, model_r = model(params, x[i], v0)
        if abs(model_l-model_r) < y_tol:
            continue
        if abs(y[i]-model_l) < y_tol or abs(y[i]-model_r) < y_tol:
            count.append(1)
        else:
            count.append(0)
    if len(count) < len(x)/4:
        return None
    else:
        return np.mean(count)


def validation_loss(out,v0,x,y):
    params = out['x']
    model_l, model_r = model(params, x, v0)
    left = np.abs(model_l - y)
    right = np.abs(model_r - y)
    tol = left<right

    #Pick the lowest loss, assume that point is parameterized by correct curve
    left_loss = np.mean(left[tol])
    right_loss = np.mean(right[np.logical_not(tol)])

    return np.mean([left_loss,right_loss])

    

def multi_regression(x_min, x_max, data, reg_func, residual = residual, n = 20, loss = 'linear', eps = 0.1, val_split = 0.2, x_scale = 1.0, plot=False, y_tol=0.04):
    val_acc = 0
    result_hold = None
    ci_deltaphi = None
    for _ in range(n):
        result, val = reg_func(x_min, x_max, data, residual = residual, loss = loss, eps = eps, val_split = val_split, x_scale=x_scale, y_tol=y_tol)
        if val is None:
            continue
        if result.success:
            if val > val_acc:
                ci_deltaphi = multi_reg_param_ci(data,x_min,x_max,result,eps)[4]
                result_hold = result
                val_acc = val
                
    if plot:
        plt.figure()
        
        i_min = search(data.x,x_min)
        i_max = search(data.x,x_max)+1
    
        x = data.x[i_min:i_max]
        y = data.y[i_min:i_max]
        
        v0 = (x_min+x_max)/2.0
        
        params = result_hold.x
        model_l, model_r = model(params, x, v0)
        plt.plot(x,y)
        plt.plot(x,model_r,c='r')
        plt.plot(x,model_l,c='k')
    return result_hold, ci_deltaphi, val_acc

def sweep_regression(data, reg_func, residual = residual, d = 6.0, c = 20, n = 20, loss = 'linear', eps = 0.1, val_split = 0.2, x_scale = 1.0, plot=False, y_tol=0.04):
    out = list()
    x_max = max(data.x) - 0.005
    x_min = min(data.x) + 0.005
    step = (x_max-x_min-2*d)/c
    ##step = (x_max-x_min)/c
    for i in range(c+1):
        #Try to start at edge with half
        center = x_min + d +  i*step
        ##center = x_min +  i*step
        low = center - d
        if low < x_min:
            low = x_min
        high = center + d
        if high > x_max:
            high = x_max
        result, ci_deltaphi, val_acc = multi_regression(low, high, data, reg_func, residual = residual, n = n, loss = loss, eps = eps, val_split = val_split, x_scale = x_scale, plot = plot, y_tol=y_tol)
        
        res_dict = {'result':result, 'x_min':low, 'x_max':high, 'eps':eps, 'ci_deltaphi':ci_deltaphi, 'val_acc':val_acc}
        
        out.append(res_dict)
    return out

def run_regression(data, reg_func, residual = residual, d = 6.0, c = 20, n = 20, loss = 'linear', eps = 0.01, val_split = 0.2, x_scale = 1.0, plot=False, y_tol=0.04):
    count = 0
    out = []
    for i in data:
        print('Running sweep ' + str(count))
        out.append(sweep_regression(i, reg_func, residual = residual, d = d, c = c, n=n, eps = eps, val_split = val_split, x_scale = x_scale, loss = loss, plot = plot, y_tol = y_tol))
        count += 1
    return out

def pick_fits(sweep_results, val_threshold = 0.7, ci_threshold = 0.001):
    hold = list()
    for i in range(len(sweep_results)):
        if sweep_results[i]['result'] is None:
            continue
        if sweep_results[i]['val_acc'] > val_threshold and 0.155 < sweep_results[i]['result'].x[4] < 1.8 and sweep_results[i]['ci_deltaphi'] < ci_threshold:
            hold.append([i,sweep_results[i]])
    out = list()
    temp = list()
    for i in range(len(hold) - 1):
        temp.append(hold[i][1])
        if hold[i][0] + 1 == hold[i+1][0]:
            continue
        else:
            index = np.array([j['ci_deltaphi'] for j in temp]).argmin()
            out.append(temp[index])
            temp = list()
    if len(temp) == 0:
        out.append(hold[-1][1])
    else:
        temp.append(hold[-1][1])
        index = np.array([j['ci_deltaphi'] for j in temp]).argmin()
        out.append(temp[index])
    return out  




def check():
    print('new2')