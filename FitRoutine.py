import numpy as np
import Analysis

#returns [fit delta phi, parity, x_loc, fit quality]
def onFit(x_data, y_data, fit, index, mindiff=0.030):
    i = index
    x = x_data[i]
    y = y_data[i]
    dy = mindiff
    if fit.pos_left <= x <= fit.pos_right:
        left = np.abs(y - fit.yValue(x)[0]) < dy
        right = np.abs(y - fit.yValue(x)[1]) < dy
        if np.abs(fit.yValue(x)[0] - fit.yValue(x)[1]) < 2 * dy:
            return [fit.delta_phi, -1, x, fit.fit_quality]
        if left and right:
            return [fit.delta_phi, -1, x, fit.fit_quality]
        if left == False and right == False:
            return [fit.delta_phi, 0, x, fit.fit_quality]
        if left:
            return [fit.delta_phi, 1, x, fit.fit_quality]
        if right:
            return [fit.delta_phi, 2, x, fit.fit_quality]
    return [-1, 0, x, 0]

def onFitSweep(x_data, y_data, fits_list, minv=1, maxv=1, mindiff=0.030, momentum = 10):

    out = list()
    if minv > 0:
        minv = x_data[0] + 0.05
    if maxv > 0:
        maxv = x_data[-1] - 0.05

    for j in fits_list:
        hold = list()
        i = 0
        while i < (len(x_data) - 1):
            if minv < x_data[i] < maxv:  # Remove the edges or open desired window
                pass
            else:
                i += 1
                continue
            of = onFit(x_data, y_data, j, i, mindiff)
            hold.append(of)
            i += 1
        out.append(np.array(hold))
    return np.array(out)

def smoothOnFit(x_data, y_data, fits_list, minv=1, maxv=1, mindiff=0.030, momentum = 10):
    ofs = np.array(onFitSweep(x_data, y_data, fits_list, minv, maxv, mindiff, momentum))
    for j in ofs:
        i = 0
        while i < len(j):
            if j[i][1] == 1 or j[i][1] == 2:
                start = i
                state = j[i][1]
                i += 1
                while True and i < len(j):
                    if j[i][0] < 0:
                        i += 1
                        break
                    if j[i][1] == state:
                        if np.abs(j[start][2] - j[i][2]) < 0.05:
                            j[start:i, 1] = state
                        break
                    elif j[i][1] == 0:
                        i += 1
                    else:
                        i += 1
                        break
            else:
                i += 1
    return ofs



# Finds edges of data for fit, returns dphi,left edge, right edge, dV of edges
def findEdge(x_data, y_data, fits_list, dV=0.025, minv=1, maxv=1, mindiff=0.030, momentum=10, prune=5):
    sof = np.array(smoothOnFit(x_data, y_data, fits_list, minv, maxv, mindiff, momentum, prune))
    out = list()
    for j in sof:
        i = 0
        hold = list()
        while i < len(j):
            state = j[i, 1]
            if state == 1 or state == 2:
                start = i
                while True:
                    i += 1
                    if i >= len(j):
                        break
                    if j[i, 1] == state:
                        continue
                    elif j[i, 0] < 0:
                        break
                    #Returns the start of the fit, and the last point, along with the dV
                    elif np.abs(j[start, 2] - j[i - 1, 2]) > dV:
                        hold.append(
                            [j[start, 0], j[start, 1], j[start, 2], j[i - 1, 2], np.abs(j[start, 2] - j[i - 1, 2]),
                             j[start, 3]])
                        break
                    else:
                        break
            else:
                i += 1
        out.append(hold)
    return out


# If a slip exists, returns the right point of edge and left point of next edge
def findSlips(x_data, y_data, fits_list, dV=0.07, minv=1, maxv=1, mindiff=0.050, minsep=0.2):
    fe = findEdge(x_data, y_data, fits_list, dV, minv, maxv, mindiff)
    out = list()
    for k in fe:
        hold = list()
        for i in range(len(k) - 1):
            j = i + 1
            if k[i][1] != k[j][1]:
                if np.abs(k[i][3] - k[j][2]) < minsep:
                    hold.append([k[i][0], k[i][1], k[i][3], k[j][2], k[i][5]])
        out.append(np.array(hold))
    return np.array(out)


def finalSlips(x_data, y_data, fits_list, dV=0.025,  minv=1, maxv=1, mindiff=0.030, minsep=0.2):
    fs = findSlips(x_data, y_data, fits_list, dV, minv, maxv, mindiff, minsep)
    alll = list()
    for i in fs:
        for j in i:
            alll.append(j)
    i = 0
    while i < len(alll):
        delete = False
        j = i + 1
        edge = alll[i][2]
        while j < len(alll):
            if np.abs(edge - alll[j][2]) < dV:
                if alll[j][4] <= alll[i][4]:
                    del alll[j]
                else:
                    del alll[i]
                    delete = True
                    j = len(alll)
            else:
                j += 1
        if not delete:
            i += 1
    return sortFinal(fits_list, alll)


def sortFinal(fits_list, fSOutput):
    alll = list(fSOutput)
    out = list()
    for j in fits_list:
        hold = list()
        fit_qual = j.fit_quality
        i = 0
        while i < len(alll):
            if alll[i][4] == fit_qual:
                hold.append(alll[i])
                del alll[i]
            else:
                i += 1
        out.append(np.array(hold))
    return np.array(out)


def countSlips(x_data, y_data, fits_list, dV=0.025, minv=1, maxv=1, mindiff=0.030, minsep=0.2):
    fs = finalSlips(x_data, y_data, fits_list, dV, minv, maxv, mindiff, minsep)
    out = list()
    for i in fs:
        for j in i:
            out.append(j[0])
    return out


# Stores the location of phase slip events, takes finalSlip and returns a list of voltage locations
# and phase values.
def countEdge(x_data, y_data, fits_list, bval, dV=0.025, minv=1, maxv=1, mindiff=0.030, minsep=0.2):
    fs = finalSlips(x_data, y_data, fits_list, dV, minv, maxv, mindiff, minsep)
    hold = list()
    n = 0
    for i in fs:
        for j in i:
            if np.abs(j[2] - j[3]) <= 2:
                hold.append([(j[2] + j[3]) / 2, bval, j[0], fits_list[n].fit_quality])
            else:
                hold.append([j[2] + 0.05, bval, j[0], fits_list[n].fit_quality])
        n += 1
    return np.array(hold)

def sortEdge(x_data, y_data, fits_list, bval, dV=0.025, minv=1, maxv=1, mindiff=0.030, minsep=0.2, boxes=[[0.15,0.35],[0.8,1.35]]):
    final = list() 
    hold = list(countEdge(x_data=x_data, y_data=y_data, fits_list=fits_list, bval=bval, dV=dV, minv=minv, maxv=maxv, mindiff=mindiff, minsep=minsep))
    for i in boxes:
        store = list()
        j = 0
        while j < len(hold):
            if hold[j][2] >= i[0] and hold[j][2] <= i[1]:
                store.append(np.array(hold.pop(j)))
            else:
                j+=1        
        final.append(np.array(store))
    final.append(np.array(hold))
    return final     




