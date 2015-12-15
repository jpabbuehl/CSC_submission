'''
Created on Nov 1, 2015

Image processing of Lightsheet acquisition
Paper: Long-term engraftment of primary bone marrow stroma promotes hematopoietic reconstitution after transplantation
Author: Jean-Paul Abbuehl

@author: Jean-Paul Abbuehl
'''
# Load depedencies
import pandas as pd
import numpy as np
import seaborn as sns
import os
import operator
from xml.etree import ElementTree as ET
from itertools import cycle

import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import *
from pylab import savefig as savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.tools.plotting import parallel_coordinates

from shapely.affinity import scale
from shapely.geometry import Point, shape, Polygon
from shapely.ops import cascaded_union

from bokeh.plotting import figure, show, output_file

import scipy.spatial as spatial
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from PIL import Image
from PIL.FontFile import WIDTH

from sklearn import mixture, metrics
from sklearn.cluster import DBSCAN

import h5py
import re
import collections as coll
import itertools as it
import javabridge as jv
import bioformats as bf

# Load second file
from registration_icp import *

def run():
    global sample, channels, segmentation, overlap_region, save_plot
    sample = 'gfp4'
    channels = 5
    overlap_region = 0.1  # 10 %

    # General parameters
    light_correction = True
    remove_duplicate_spot = True
    registration = 'simple'  # simple or icp
    save_plot = True  # Save plot instead of showing

    # Sample specific parameters
    # Gating list specifies the order of gating
    # Channel 1 = Lineage, Channel 2 = Sca1, Channel 3 = GFP, Channel 4 =
    # 7AAD, Channel 5 = SLAM

    # Logic list specifies the constraint for the gating
    # Logic 0 = Strickly negative, Logic 1 = Negative, Logic 2 = Positive,
    # Logic 3 = Strickly positive
    HSC_gating = [3, 1, 5, 2, 4]
    HSC_logic = [1, 0, 3, 2, 2]
    SSC_gating = [3, 2]
    SSC_logic = [3, 3]
    PROG_gating = [3, 2]
    PROG_logic = [3, 1]

    # DBscan parameters
    min_samples_per_cluster = 4
    threshold_distance = [40, 200]  # min and high

    # Cache for loading preprocessed data, if already computed before
    cache_ROI = False
    cache_cells = False

    # Graphical parameters
    global plot_size, marker_size
    plot_size = 10
    marker_size = 6
    graph_roi = True
    graph_duplicate = False
    graph_classifyier = False
    graph_map = True
    graph_cluster = False
    ROI_extraction = False  # not implemented yet

    # Define global variable
    global nrow, ncol, pixelSize, pixelDepth, pixelFormat, keys, xml, image, scaleX, scaleY
    segmentation = "data//" + sample + "//segmentation//"
    xml = "data//" + sample + "//" + sample + ".mvl"
    image = "data//" + sample + "//" + sample + " overview.tif"
    ncol, nrow, pixelFormat = XML_process(xml)
    pixelSize = 0.3218071
    pixelDepth = 1.09
    keys = ['VIEW', 'ID', 'X', 'Y', 'Z', 'QUALITY']
    for c in xrange(1, channels + 1):
        prefix = 'CH' + str(c) + '_'
        channel_key = [prefix + 'AREA', prefix + 'INTENSITY',
                       prefix + 'MIN', prefix + 'MAX', prefix + 'STD']
        keys = keys + channel_key
    im = mpimg.imread(image)
    scaleX = ncol * pixelFormat / im.shape[1] * pixelSize
    scaleY = nrow * pixelFormat / im.shape[0] * pixelSize

    # Start processing
    if cache_ROI:
        polygon = np.load("data//" + sample + "//POLY.npy")
        polygon = Polygon(zip(polygon[:, 0], polygon[:, 1]))
    else:
        polygon, scaleX, scaleY = ROI_define()
        x, y = polygon.exterior.xy
        xypoly = np.column_stack((np.array(x), np.array(y)))
        np.save("data//" + sample + "//POLY.npy", xypoly)
        # Save as CSV for multitype statistic test with spatstats
        np.savetxt("data//" + sample + "//POLY.csv", xypoly, delimiter=',')

    if cache_cells:
        Fsuffix = "data//" + sample + "//HSC" + suffix(HSC_gating, HSC_logic)
        HSC_result = np.load(Fsuffix + '.npy')
        Fsuffix = "data//" + sample + "//SSC" + suffix(SSC_gating, SSC_logic)
        SSC_result = np.load(Fsuffix + '.npy')
        Fsuffix = "data//" + sample + "//PROG" + \
            suffix(PROG_gating, PROG_logic)
        PROG_result = np.load(Fsuffix + '.npy')
    else:
        # Loading and correcting data
        data = DATA_loading(light_correction)
        data = ROI_filtering(data, polygon, graph_roi)
        spot_duplicate_tolerance = 5.0  # Maximum distance for duplicate detection
        data = DUPLICATE_remove(
            data, spot_duplicate_tolerance, graph_duplicate)

        # Classifier
        HSC_result = CELL_classifier(
            data, HSC_gating, HSC_logic, graph_classifyier, graph_map)
        Fsuffix = "data//" + sample + "//HSC" + suffix(HSC_gating, HSC_logic)
        np.save(Fsuffix + '.npy', HSC_result)
        np.savetxt(Fsuffix + '.csv', HSC_result, delimiter=',')

        SSC_result = CELL_classifier(
            data, SSC_gating, SSC_logic, graph_classifyier, graph_map)
        Fsuffix = "data//" + sample + "//SSC" + suffix(SSC_gating, SSC_logic)
        np.save(Fsuffix + '.npy', SSC_result)
        np.savetxt(Fsuffix + '.csv', SSC_result, delimiter=',')

        PROG_result = CELL_classifier(
            data, PROG_gating, PROG_logic, graph_classifyier, graph_map)
        Fsuffix = "data//" + sample + "//PROG" + \
            suffix(PROG_gating, PROG_logic)
        np.save(Fsuffix + '.npy', PROG_result)
        np.savetxt(Fsuffix + '.csv', PROG_result, delimiter=',')

    # Distance calculation
    print 'HSC detected:%d' % HSC_result.shape[0]
    print 'SSC detected:%d' % SSC_result.shape[0]
    print 'PROG detected:%d' % PROG_result.shape[0]

    distance1, index1 = DISTANCE_spatial(SSC_result, HSC_result)
    np.savetxt("data//" + sample + "//distance_SSC_output.csv",
               distance1, delimiter=',')

    distance2, index2 = DISTANCE_spatial(PROG_result, HSC_result)
    np.savetxt("data//" + sample + "//distance_PROG_output.csv",
               distance2, delimiter=',')

    # Violin plot comparing HSC-HSC distance, in respect of SSC proximity
    df = pd.DataFrame({'distance': np.concatenate((distance1, distance2)),'sample': ([1] * (len(distance1) + len(distance2)))})
    df['population']=['SSC'] * len(distance1) + ['PROG'] * len(distance2)
    sns.set_style("whitegrid")
    sns.plt.grid(False)
    sns.violinplot(x="sample", y="distance", hue="population", data=df, palette="Set2",
                   split=False, scale="area", cut=0, inner="quartile", orient='v')
    plt.title('Distance between HSC and SSC / Progenitors')
    sns.despine()
    sns.plt.show()

    # Extract All ROI with interaction HSC and SSC, not yet implemented
    if ROI_extraction:
        close_mask=distance1 < threshold_distance[0]
        SSC_subset=SSC_result[close_mask, :]
        coordinates=ROI_definition(SSC_subset)
        diameter=200.0
        ROI_crop(coordinates, diameter)

    # DBscan clustering
    distance3, index3=DISTANCE_spatial(HSC_result, SSC_result)
    close_mask=distance3 < threshold_distance[0]
    HSC_close=HSC_result[close_mask, :]
    far_mask=distance3 > threshold_distance[1]
    HSC_far=HSC_result[far_mask, :]

    within_cluster_threshold=threshold_distance[0]
    title1='Clustering of HSC close to SSC - %f' % threshold1
    ClHSC_close=cDBScan(HSC_close[:, [2, 3]], title1, within_cluster_threshold,
                          graph_cluster, min_samples_per_cluster, plot_size, marker_size)
    title2='Clustering of HSC far from SSC - %f' % threshold2
    ClHSC_far=cDBScan(HSC_far[:, [2, 3]], title2, within_cluster_threshold,
                        graph_cluster, min_samples_per_cluster, plot_size, marker_size)

    Cluster_plot(ClHSC_close, ClHSC_far, min_samples_per_cluster, threshold_distance, [
                 HSC_close.shape[0], HSC_far.shape[0]], save_plot)

def suffix(gating, logic):
    output=''
    for i in xrange(len(gating)):
        output=output + '_' + str(gating[i]) + '-' + str(logic[i])
    return output


def ROI_define():
    im=mpimg.imread(image)
    height=im.shape[0] / nrow
    width=im.shape[1] / ncol

    # Axis adjustable in function of views during acquisition
    fig=plt.figure(figsize=(10, 10 * height / width))
    plt.imshow(im)
    plt.grid(True)
    plt.xticks([width * i for i in range(1, ncol)])
    plt.yticks([height * i for i in range(1, nrow)])

    # Add mouse click event to set area of interest
    ax2=fig.add_subplot(111)
    ax2.patch.set_alpha(0.5)
    cnv=Canvas(ax2)
    plt.connect('button_press_event', cnv.update_path)
    plt.connect('motion_notify_event', cnv.set_location)
    plt.show()

    # Deal with polygon
    poly=cnv.extract_poly()
    poly, scaleX, scaleY=coordinate_conversion(
        poly, im.shape[1], im.shape[0])
    return poly, scaleX, scaleY


def DUPLICATE_remove(data, distance_threshold, graph=False):
    sizeFormat=pixelFormat * pixelSize
    poly=overlapping_grid(nrow, ncol, overlap_region,
                            [sizeFormat, sizeFormat])
    result=np.zeros(data.shape[0], dtype=bool)
    for i in xrange(data.shape[0]):
        result[i]=poly.contains(Point(data[i, 2], data[i, 3]))
    data_to_evaluate=data[result.ravel(), :]
    output=data[np.invert(result.ravel()), :]
    # Calculate Nearest Neighbors distance
    distance, indexes=DISTANCE_spatial(
        data_to_evaluate, data_to_evaluate, neighbors=2)
    if graph:
        sns.plt.grid(False)
        sns.distplot(distance)
        plt.title("%d spots to evaluate" % data_to_evaluate.shape[0])
        sns.despine()
        sns.plt.show()
    # Find which spots is below the distance_threshold
    mask_to_keep=distance > distance_threshold
    to_keep=[]
    to_exclude=[]
    for source in xrange(len(indexes)):
        target=indexes[source]
        if indexes[target] == source and distance[source] < distance_threshold and source not in to_exclude:
            to_keep.append(source)
            to_exclude.append(target)
    mask_to_keep[to_keep]=True
    print str(len(to_exclude)) + ' spots discarded'
    data_to_evaluate=data_to_evaluate[mask_to_keep, :]
    output=np.row_stack((output, data_to_evaluate))
    return output


def CELL_plotting(data, channel, markersize, limit=True, top_title='title'):
    # Limit nb scatter point for performance reason:
    if limit:
        data=data[np.random.choice(data.shape[0], 10000), :]
    if type(channel) is list:
        row=1
        column=len(channel)
        fig, ax=plt.subplots(row, column, facecolor='w', figsize=(15, 10))
        fig.suptitle(top_title, fontsize=24)
        plt.grid(False)
        im=mpimg.imread(image)
        gray=rgb2gray(im)
        # iterates over each axis, ax, and plots random data
        for i, ax in enumerate(ax.flat, start=1):
            ax.set_title('Spots: channel' + str(channel[i - 1]))
            ax.imshow(gray, cmap=plt.get_cmap('gray'))
            points=convertXYforPlot(
                data[:, [2, 3]], pixelSize, scaleX, scaleY)
            colindex=DATA_colIndex('INTENSITY', channel[i - 1])
            img=ax.scatter(points[:, 0], points[:, 1], c=data[:, colindex], marker='o', s=np.repeat(
                markersize, data.shape[0]), alpha=0.75, cmap=cm.hsv)
            div=make_axes_locatable(ax)
            cax=div.append_axes("right", size="15%", pad=0.05)
            cbar=plt.colorbar(img, cax=cax)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        fig=plt.figure()
        plt.grid(False)
        im=mpimg.imread(image)
        gray=rgb2gray(im)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        points=convertXYforPlot(data[:, [2, 3]], pixelSize, scaleX, scaleY)
        colindex=DATA_colIndex('INTENSITY', channel)
        scatter(points[:, 0], points[:, 1], c=data[:, colindex], marker='o', s=np.repeat(
            markersize, data.shape[0]), alpha=0.75, cmap=cm.hsv)
        plt.colorbar()
        title='Spots: channel' + str(channel)
        plt.title(title)
        # Formating graph
        plt.show()


def ROI_filtering(data, poly, graph):
    result=np.zeros(data.shape[0], dtype=bool)
    print str(data.shape[0]) + ' objects before filtering'
    if graph:
        im=mpimg.imread(image)
        height=im.shape[0]
        width=im.shape[1]
        ROI_plot(data, poly, 'before filtering', 5.0, 5, width, height)
    for i in xrange(data.shape[0]):
        result[i]=poly.contains(Point(data[i, 2], data[i, 3]))
    data=data[result.ravel(), :]
    print str(data.shape[0]) + ' objects after filtering'
    if graph:
        ROI_plot(data, poly, 'after filtering', 5.0, 5, width, height)
    return data


def convertXYforPlot(data, pixelSize, scaleX, scaleY):
    data[:, 0]=(data[:, 0] / scaleX)
    data[:, 1]=(data[:, 1] / scaleY)
    data=data.astype(int)
    return data


def coordinate_conversion(data, width, height):
    scaleX=ncol * pixelFormat / width * pixelSize
    scaleY=nrow * pixelFormat / height * pixelSize
    Xmap=map(int, data[:, 0].tolist())
    Ymap=map(int, data[:, 1].tolist())
    geom=Polygon(zip(Xmap, Ymap))
    geom2=scale(geom, xfact=scaleX, yfact=scaleY, origin=((0, 0)))
    return geom2, scaleX, scaleY


class Canvas(object):

    def __init__(self, ax):
        self.ax=ax
        # Set limits to unit square
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(top=0)
        # turn off axis
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        # Create handle for a path of connected points
        self.path, = ax.plot([], [], 'ro-', lw=2)
        self.vert=[]
        self.ax.set_title(
            'LEFT: new point, MIDDLE: delete last point, RIGHT: close polygon')
        self.x=[]
        self.y=[]
        self.mouse_button={1: self._add_point,
                             2: self._delete_point, 3: self._close_polygon}
        self.flag_close=False

    def set_location(self, event):
        if event.inaxes:
            self.x=event.xdata
            self.y=event.ydata

    def _add_point(self):
        self.vert.append((self.x, self.y))

    def _delete_point(self):
        if len(self.vert) > 0:
            self.vert.pop()

    def _close_polygon(self):
        self.vert.append(self.vert[0])
        self.flag_close=True

    def update_path(self, event):
        # If the mouse pointer is not on the canvas, ignore buttons
        if not event.inaxes:
            return
        # Do whichever action correspond to the mouse button clicked
        self.mouse_button[event.button]()
        x=[self.vert[k][0] for k in range(len(self.vert))]
        y=[self.vert[k][1] for k in range(len(self.vert))]
        self.path.set_data(x, y)
        if self.flag_close:
            plt.close()
        else:
            plt.draw()

    def extract_poly(self):
        return np.array(self.vert)


def DATA_loading(light_correction,registration):
    data=read_views(registration)
    data=distance_correction(data)
    if light_correction:
        data=illumination_correction(data, percentile)
    return data


def CELL_classifier(data, gating, logic, plotting, map_plotting):
    remaining=data
    for i in xrange(len(gating)):
        index=DATA_colIndex('INTENSITY', gating[i])
        title='Classification channel' + str(gating[i])
        Cmean, Cweight, Cclass=fit_mixture(
            remaining[:, [index]], 3, title, plotting, logic[i])
        min_index, min_value=min(
            enumerate(Cmean), key=operator.itemgetter(1))
        max_index, max_value=max(
            enumerate(Cmean), key=operator.itemgetter(1))
        Cclass=format_class(Cclass, min_index, max_index, logic[i])
        remaining=remaining[Cclass, :]
    if map_plotting:
        s1=np.char.array(data[:, 0]) + '-' + np.char.array(data[:, 1]) + \
            '-' + np.char.array(data[:, 2]) + '-' + np.char.array(data[:, 3])
        s2=np.char.array(remaining[:, 0]) + '-' + np.char.array(remaining[:, 1]) + \
            '-' + np.char.array(remaining[:, 2]) + \
            '-' + np.char.array(remaining[:, 3])
        idx=np.where(np.in1d(s1, s2))[0]
        bool_detect=np.repeat(False, data.shape[0])
        bool_detect[idx]=True
        # Parallel Coordinates with PANDAS
        # parallel_coordinates_plot(data,gating,bool_detect) # working ok but
        # need performance optimization
        CELL_plotting(remaining, gating, 15, False,
                      "%d cells detected in total" % remaining.shape[0])
    return remaining


def DISTANCE_frequency(data1, data2, length):
    width=ncol * pixelFormat * pixelSize
    height=nrow * pixelFormat * pixelSize
    win_x=int(width // length)
    win_y=int(height // length)
    for i in xrange(win_x):
        for j in xrange(win_y):
            x1=length * float(i)
            x2=length * (float(i) + 1.0)
            y1=length * float(j)
            y2=length * (float(j) + 1.0)
            D1x=data1[(x2 > data1[:, 2]) & (data1[:, 2] > x1), :]
            D1y=data1[(y2 > data1[:, 3]) & (data1[:, 3] > y1), :]
            if D1x.shape[0] > 0 and D1y.shape[0] > 0:
                kdtree1=cKDTree(D1x[:, [2, 3]])
                dists1, inds1=kdtree1.query(
                    D1y[:, [2, 3]], distance_upper_bound=1e-5)
                count1=(dists1 == 0).sum()
            else:
                count1=0

            D2x=data2[(x2 > data2[:, 2]) & (data2[:, 2] > x1), :]
            D2y=data2[(y2 > data2[:, 3]) & (data2[:, 3] > y1), :]
            if D2x.shape[0] > 0 and D2y.shape[0] > 0:
                kdtree2=cKDTree(D2x[:, [2, 3]])
                dists2, inds2=kdtree2.query(
                    D2y[:, [2, 3]], distance_upper_bound=1e-5)
                count2=(dists2 == 0).sum()
            else:
                count2=0

            if 'result' not in locals() and count1 != 0:
                result=[[count1, count2]]
            elif count1 != 0:
                result.append([count1, count2])
    result=np.array(result)
    return result


def distance_correction(data):
    data[:, 2]=data[:, 2] * pixelSize
    data[:, 3]=data[:, 3] * pixelSize
    data[:, 4]=data[:, 4] * pixelDepth
    return data

# Calculate Nearest distance from each point in data1 to any point in data2


def DISTANCE_spatial(data1, data2, neighbors=1):
    cloud1=data1[:, [2, 3, 4]]
    cloud2=data2[:, [2, 3, 4]]
    distance, index=spatial.KDTree(cloud2).query(cloud1, k=neighbors)
    if neighbors > 1:
        distance=distance[:, [(neighbors - 1)]].ravel()
        index=index[:, [(neighbors - 1)]].ravel()
    index=index.astype(int)
    return distance, index


def DATA_colIndex(parameter, channel):
    to_find='CH' + str(channel) + '_' + parameter
    return keys.index(to_find)

# Get views and compare MFIs of positive nucleus population


def illumination_correction(data, percentile):
    index=DATA_colIndex('INTENSITY', channels - 1)
    view_nb=ncol * nrow
    correction_factor_min=np.ones(view_nb)
    correction_flag=np.repeat(False, view_nb)
    for i in xrange(0, view_nb):
        subdata=data[data[:, 0] == (1 + i), :]
        subdata=subdata[:, [index]]
        if subdata.shape[0] > 500:
            correction_flag[i]=True
            Cmean, Cweight, Cclass=fit_mixture(
                subdata, 3, 'illumination correction view %d' % i, False, 3)
            print 'view' + str(i + 1) + ' / Centroid Mean: [%s]' % ', '.join(map(str, Cmean))
            min_index, min_value=min(
                enumerate(Cmean), key=operator.itemgetter(1))
            correction_factor_min[i]=min_value
    correction_median=np.median(correction_factor_min[correction_flag])
    correction_factor_min[correction_flag]=correction_factor_min[
        correction_flag] / correction_median
    for i in xrange(0, view_nb):
        subdata=data[data[:, 0] == (1 + i), :]
        for channel in xrange(1, channels + 1):
            index=DATA_colIndex('INTENSITY', channel)
            subdata[:, [index]]=subdata[
                :, [index]] * correction_factor_min[i]
        if i == 0:
            corrected_data=subdata
        else:
            corrected_data=np.row_stack((corrected_data, subdata))
    if data.shape == corrected_data.shape:
        return corrected_data
    else:
        print 'error corrected array has not same size as original'
        raise


def ROI_plot(data, poly, title, pltsize, size, xlim, ylim):
    x, y=poly.exterior.xy
    fig=plt.figure(figsize=(int(pltsize), int(pltsize * ylim / xlim)))
    axes=plt.gca()
    axes.set_xlim([0, xlim * scaleX])
    axes.set_ylim([0, ylim * scaleY])
    # plot points in 2D
    class1=data.astype(int)
    if class1.shape[0] > 30000:
        class1=class1[np.random.choice(class1.shape[0], 30000), :]
    scatter(class1[:, 2], class1[:, 3], c='r', marker='o', s=5, alpha=0.75)
    ax=fig.add_subplot(111)
    ax.plot(x, y, color='#6699cc', alpha=0.7,
            linewidth=3, solid_capstyle='round', zorder=2)
    # Formating graph
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()


def format_class(Cclass, min_index, max_index, logic):
    if logic == 3:
        output=(Cclass == max_index)
    elif logic == 2:
        output=(Cclass != min_index)
    elif logic == 1:
        output=(Cclass != max_index)
    elif logic == 0:
        output=(Cclass == min_index)
    else:
        print 'error logic formating'
        raise
    return output


def index_min(values):
    return min(xrange(len(values)), key=values.__getitem__)


def fit_mixture(data, ncomp, title, doplot, logic):
    # init_params=mean0 500 mean1 1500
    clf=mixture.GMM(n_components=ncomp,
                      covariance_type='full', init_params='wmc')
    clf.fit(data)
    ml=clf.means_
    wl=clf.weights_
    cl=clf.covars_
    ms=[m[0] for m in ml]
    ws=[w for w in wl]
    classes=clf.predict(data)
    if doplot == True:
        min_index, min_value=min(enumerate(ml), key=operator.itemgetter(1))
        max_index, max_value=max(enumerate(ml), key=operator.itemgetter(1))
        mid_index=len(ml) - min_index - max_index
        histdist=plt.hist(data, 100, normed=True)
        if logic == 0:
            Scolor=['g', 'r', 'r']
        elif logic == 1:
            Scolor=['g', 'g', 'r']
        elif logic == 2:
            Scolor=['r', 'g', 'g']
        elif logic == 3:
            Scolor=['r', 'r', 'g']
        plotgauss1=lambda x: plt.plot(x, wl[min_index] * matplotlib.mlab.normpdf(
            x, ml[min_index], np.sqrt(cl[min_index]))[0], linewidth=5, color=Scolor[0])
        plotgauss2=lambda x: plt.plot(x, wl[mid_index] * matplotlib.mlab.normpdf(
            x, ml[mid_index], np.sqrt(cl[mid_index]))[0], linewidth=5, color=Scolor[1])
        plotgauss3=lambda x: plt.plot(x, wl[max_index] * matplotlib.mlab.normpdf(
            x, ml[max_index], np.sqrt(cl[max_index]))[0], linewidth=5, color=Scolor[2])
        plotgauss1(histdist[1])
        plotgauss2(histdist[1])
        plotgauss3(histdist[1])
        plt.title(title)
        plt.show()
    return ms, ws, classes


def read_views(registration):
    # Get all CSV
    infiles=os.listdir(segmentation)
    view_nb=int(len(infiles) / channels)

    # Get all transition
    if 'simple' in registration:
        Xregister, Yregister=simple_merge()
    elif 'icp' in registration:
        Xregister, Yregister=icp()
    else:
        print 'registration should be either simple or icp'
        raise()
    for i in xrange(1, view_nb + 1):
        print i
        data=read_view(i)
        if data is None:
            # Empty
            continue
        data[:, 2]=data[:, 2] + Xregister[i - 1]
        data[:, 3]=data[:, 3] + Yregister[i - 1]
        if i == 1:
            output=data
        else:
            output=np.row_stack((output, data))
    return output


def read_view(view):
    flag=True
    for c in xrange(1, channels + 1):
        infile=segmentation + '//ch' + str(c) + 'view' + str(view) + '.csv'
        df=pd.read_csv(infile, sep=',')
        data=np.array(df.values)
        # make array absolute positive values
        data=np.fabs(data)
        # Sort by spot ID
        data=data[data[:, 0].argsort()]
        col=[5, 6, 7, 8, 9]
        if flag:
            output=data
            viewCol=np.repeat(view, output.shape[0]).reshape(-1, 1)
            output=np.hstack((viewCol, output))
            flag=False
            IDcheck=data[:, 0]
        else:
            np.testing.assert_array_equal(IDcheck, data[:, 0])
            output=np.column_stack((output, data[:, col]))
    if output.shape[0] > 0:
        mask=np.isnan(output)
        mask=np.invert(mask)
        mask=np.all(mask, axis=1)
        output=output[mask, :]
        return output
    else:
        return None


def XML_process(infile):
    tree=ET.parse(infile)
    root=tree.getroot()
    data=root[1]

    nb_views=len(data)
    positionX=[]
    positionY=[]
    PixFormat=float(data[0].attrib['AcquisitionFrameWidth'])
    for i in xrange(0, nb_views):
        positionX.append(float(data[i].attrib['PositionX']))
        positionY.append(float(data[i].attrib['PositionY']))
    positionX=[x - min(positionX) for x in positionX]
    positionY=[y - min(positionY) for y in positionY]
    positionX=[max(positionX) - x for x in positionX]

    Xview_nb=len(list(set(positionX)))
    Yview_nb=len(list(set(positionY)))
    return (Xview_nb, Yview_nb, PixFormat)


def snake_generator(Ncol, Nrow):
    # Snake X shift, start top left
    Xsnake=[]
    view_id=1
    for y in xrange(1, Nrow + 1):
        if (view_id > Ncol * Nrow + 1):
            break
        if(y % 2 != 0):
            # Compute from left to right
            Xsnake=Xsnake + ([i for i in xrange(view_id, Ncol + view_id)])
            view_id=max(Xsnake)
        else:
            # Compute from right to left
            Xsnake=Xsnake + \
                ([i for i in xrange(Ncol + view_id, view_id, -1)])
            view_id=max(Xsnake) + 1
    adjust_Xsnake=[]
    for y in xrange(0, Nrow + 1):
        start=Ncol * y
        end=start + Ncol
        if end < Ncol * Nrow + 1:
            adjust_Xsnake.append(Xsnake[start:end])
    Xsnake=adjust_Xsnake
    # Snake Y shift, start top right
    Ysnake=[]
    ScrollY=[]
    for i in xrange(1, Nrow + 1):
        if(i % 2 != 0):
            ScrollY.append(i)
    for x in xrange(0, Ncol):
        for y in ScrollY:
            if((Ncol * y - x) <= (Nrow * Ncol)):
                Ysnake.append(Ncol * y - x)
            if((Ncol * y + x + 1) <= (Nrow * Ncol)):
                Ysnake.append(Ncol * y + x + 1)
    # Adjust to start from top left
    adjust_Ysnake=[]
    for i in xrange(Ncol - 1, -1, -1):
        start=(Nrow) * i
        if(start < 0):
            break
        end=start + Nrow
        adjust_Ysnake.append(Ysnake[start:end])
    Ysnake=adjust_Ysnake
    return (Xsnake, Ysnake)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def simple_merge():
    Ncol, Nrow, totalPix=XML_process(xml)
    Xsnake, Ysnake=snake_generator(Ncol, Nrow)
    npXsnake=np.array(Xsnake)
    npYsnake=np.array(Ysnake)
    Xoutput=np.zeros(np.amax(npXsnake))
    Youtput=np.zeros(np.amax(npYsnake))
    Sformat=totalPix
    for row in xrange(0, npXsnake.shape[0]):
        Yposition=Sformat * row * (1.0 - overlap_region)
        for col in xrange(0, npXsnake.shape[1]):
            view=npXsnake[row][col]
            Xposition=Sformat * col * (1.0 - overlap_region)
            Xoutput[view - 1]=Xposition
            Youtput[view - 1]=Yposition
    return Xoutput, Youtput


def overlapping_grid(nrow, ncol, overlap, format):
    # Strategy, make several horizontal and vertical bar, and make union of all of them
    # Format is xy dimension of a view, 0 is x, 1 is y
    overlaping_step=format[0] * overlap
    max_x=format[0] * ncol - (ncol - 1) * overlaping_step * 2.0
    max_y=format[1] * nrow - (nrow - 1) * overlaping_step * 2.0
    total_polygons=[]
    for x in xrange(1, int(ncol)):
        xcenter=format[0] * x - overlaping_step * (x - 1) * 2.0
        area=Polygon([(xcenter - overlaping_step, 0.0),
                        (xcenter - overlaping_step, max_y),
                        (xcenter + overlaping_step, max_y),
                        (xcenter + overlaping_step, 0.0)])
        total_polygons.append(area)
    for y in xrange(1, int(nrow)):
        ycenter=format[1] * y - overlaping_step * (y - 1) * 2.0
        area=Polygon([(0, ycenter - overlaping_step),
                        (max_x, ycenter - overlaping_step),
                        (max_x, ycenter + overlaping_step),
                        (0, ycenter + overlaping_step)])
        total_polygons.append(area)
    grid=cascaded_union(total_polygons)
    return grid


def trial_overlaping():
    # im = mpimg.imread(image)
    # height = im.shape[0]
    # width = im.shape[1]
    # ROI_plot(poly, 'before filtering', 5.0, 5, width, height)
    #     global scaleX, scaleY
    #     im = mpimg.imread(image)
    #     scaleX = ncol * pixelFormat / im.shape[1] * pixelSize
    #     scaleY = nrow * pixelFormat / im.shape[0] * pixelSize
    from descartes import PolygonPatch
    from matplotlib.patches import Polygon
    import pylab as pl
    BLUE='#6699cc'
    GRAY='#999999'
    poly=overlapping_grid(10.0, 6.0, 0.1, [1920.0, 1920.0])
    fig, ax=pl.subplots()
    patch2b=PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    ax.add_patch(patch2b)
    ax.autoscale_view(tight=True)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()

# Find optimal EPS or MIN-samples in fct of the data


def estimate_param_DBScan(data, eps, min_samples):
    max_score=0.0
    cluster_result=[]
    if min_samples.shape[0] == 1:
        for i in xrange(len(eps)):
            db=DBSCAN(eps[i], min_samples).fit(data)
            core_samples_mask=np.zeros_like(db.labels_, dtype=bool)
            # Determine which cells were clustered
            core_samples_mask[db.core_sample_indices_]=True
            labels=db.labels_
            n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ > 1:
                score=metrics.silhouette_score(data, labels)
                cluster_result.append(n_clusters_)
                print('EPS: %f' % eps[i])
                print('Estimated number of clusters: %d' % n_clusters_)
                print("Silhouette Coefficient: %0.3f" % score)
            else:
                cluster_result.append(0)
        data=pd.DataFrame({'cluster': cluster_result, 'eps': eps})
        data.plot('eps', 'cluster', kind='line')
        plt.show()
    elif eps.shape[0] == 1:
        for j in xrange(len(min_samples)):
            db=DBSCAN(eps, min_samples[j]).fit(data)
            core_samples_mask=np.zeros_like(db.labels_, dtype=bool)
            # Determine which cells were clustered
            core_samples_mask[db.core_sample_indices_]=True
            labels=db.labels_
            n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ > 1:
                score=metrics.silhouette_score(data, labels)
                cluster_result.append(n_clusters_)
                print('minPTS: %d' % min_samples[j])
                print('Estimated number of clusters: %d' % n_clusters_)
                print("Silhouette Coefficient: %0.3f" % score)
            else:
                cluster_result.append(0)
        data=pd.DataFrame({'cluster': cluster_result, 'minPTS': min_samples})
        data.plot('minPTS', 'cluster', kind='line')
        plt.show()
    else:
        best_eps=0
        best_min_samples=0
        for i in xrange(len(eps)):
            for j in xrange(len(min_samples)):
                db=DBSCAN(eps[i], min_samples[j]).fit(data)
                core_samples_mask=np.zeros_like(db.labels_, dtype=bool)
                # Determine which cells were clustered
                core_samples_mask[db.core_sample_indices_]=True
                labels=db.labels_
                n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_ > 1:
                    score=metrics.silhouette_score(data, labels)
                    print('EPS: %f' % eps[i])
                    print('minPTS: %d' % min_samples[j])
                    print('Estimated number of clusters: %d' % n_clusters_)
                    print("Silhouette Coefficient: %0.3f" % score)
                    if score > max_score:
                        max_score=score
                        best_eps=eps[i]
                        best_min_samples=min_samples[j]
        print('Best EPS: %f' % best_eps)
        print('Best minPTS: %d' % best_min_samples)


def cDBScan(data, title, eps, graph_cluster, min_samples, pltsize, msize):
    # No normalization, because it distort distances
    db=DBSCAN(eps, min_samples).fit(data)
    core_samples_mask=np.zeros_like(db.labels_, dtype=bool)
    # Determine which cells were clustered
    core_samples_mask[db.core_sample_indices_]=True
    labels=db.labels_
    n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)
    score=metrics.silhouette_score(data, labels)

    if graph_cluster:
        im=mpimg.imread(image)
        plt.figure(figsize=(int(pltsize), int(
            pltsize * im.shape[0] / im.shape[1])))
        plt.grid(False)
        axes=plt.gca()
        axes.set_xlim([0, int(im.shape[1] * scaleX)])
        axes.set_ylim([int(im.shape[0] * scaleY), 0])
        img=Image.open(image)
        rsize=img.resize(
            (int(img.size[0] * scaleX), int(img.size[1] * scaleY)))
        im=np.asarray(rsize)
        gray=rgb2gray(im)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
    # Black removed and is used for noise instead.
    unique_labels=set(labels)
    colors=plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    core=[]
    edges=[]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col='k'

        class_member_mask=(labels == k)
        xy=data[class_member_mask & core_samples_mask]
        if k != -1:
            core.append(xy.shape[0])
        if graph_cluster:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=msize)

        xy=data[class_member_mask & ~core_samples_mask]
        if k != -1:
            edges.append(xy.shape[0])
        if graph_cluster:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=int(msize / 2))
    if graph_cluster:
        plt.suptitle(title, fontsize=12)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    cluster_size=map(add, core, edges)
    return cluster_size


def Cluster_plot(v1, v2, min_bin, min_eps, distances, total_HSC, save_plot):
    bins=np.linspace(min_bin, max(v1 + v2), int(max(v1 + v2) / 2))
    # V1 length clusters
    # V2 length clusters
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.suptitle('Cluster size: %d total HSC' %
                 sum(total_HSC), fontsize=14, fontweight='bold')
    info_title="Close HSC clusters: %d (%d um)   Far HSC cluster: %d (%d um)" % (
        len(v1), distances[0], len(v2), distances[1])
    ax.set_title(info_title)
    ax.set_xlabel('Cells per cluster')
    ax.set_ylabel('Nb of cluster')
    c1=int(sum(v1) / float(total_HSC[0]) * 100)
    c2=int(sum(v2) / float(total_HSC[1]) * 100)
    sns.plt.grid(False)
    plt.hist(v1, bins, alpha=0.5, label='Close HSC clusters: ' +
             ' - ' + str(c1) + '%')
    plt.hist(v2, bins, alpha=0.5, label='Far HSC clusters: ' +
             ' - ' + str(c2) + '%')
    plt.legend(loc='upper right')
    sns.despine()
    if save_plot:
        suffix=str(min_bin) + 'pts_' + str(min_eps) + 'perc'
        savefig("data//" + sample + "//clustering_" + suffix + ".png")
    else:
        plt.show()


def parallel_coordinates_plot(data, gating, interest):
    from pandas.tools.plotting import parallel_coordinates
    # Subset data for gating only
    indices=[]
    index=[]
    for i in xrange(len(gating)):
        indices.append('CH' + str(gating[i]))
        index.append(DATA_colIndex('INTENSITY', gating[i]))
    data=data[:, index]
    data=pd.DataFrame(data, columns=indices)
    data['interest']=interest
    data.to_pickle('panda.pkl')
    data=data.apply(normalize)
    pd.parallel_coordinates(data, 'interest')
    sns.plt.grid(False)
    sns.despine()
    plt.show()


def normalize(df, from_joint):
    df.drop(['interest', from_joint], axis=1,
            level='joint').sub(df[from_joint], level=1)

def ROI_definition(data):
    views = np.unique(data[:,[0]].ravel())
    points = data[:,[1]].ravel()
    # Read original files to extract X,Y,Z
    X=np.zeros(data.shape[0])
    Y=X
    Z=X
    ID_VIEW=X
    i=0
    for view in views:
        data=read_view(view)
        for point in points:
            pt_data=data[data[:,1]==point,:]
            X[i]=int(pt_data[:,DATA_colIndex('X',)] / pixelSize)
            Y[i]=int(pt_data[:,DATA_colIndex('Y',)] / pixelSize)
            Z[i]=int(pt_data[:,DATA_colIndex('Z',)] / pixelDepth)
            ID_VIEW[i]=view
            i+=1
    return  np.column_stack((ID_VIEW,X,Y,Z))
    

# incomplete, need to specify range for loading and save subregion
def ROI_crop(coordinates, diameter):
    # Call JavaBridge Bioformat Plugin
    VM_STARTED = False
    VM_KILLED = False
    DEFAULT_DIM_ORDER = 'XYZTC'
    
    BF2NP_DTYPE = {
        0: np.int8,
        1: np.uint8,
        2: np.int16,
        3: np.uint16,
        4: np.int32,
        5: np.uint32,
        6: np.float32,
        7: np.double
    }
    # Start Java virtual machine
    JVstart()
    for point in coordinates:
        filename = 'view'+str(point[0])+'ids'
        md = bf.get_omexml_metadata(filename)
        # Parse XML string into XML object
        xml = ET.ElementTree(ET.fromstring(md))
        metadata=parse_xml_metadata_IDS(xml)
        rdr = bf.ImageReader(filename, perform_init=True)
        image=np.empty(metadata['format'][2])
        z=100
        a=rdr.read(z=z,c=1)
        plt.imshow(a,cmap=cm.gray)
        plt.show()
    
    # End Java virtual machine
    JVdone()

def JVstart(max_heap_size='8G'):
    """Start the Java Virtual Machine, enabling bioformats IO.
    max_heap_size : string, optional
        The maximum memory usage by the virtual machine. Valid strings
        include '256M', '64k', and '2G'. Expect to need a lot.
    """
    jv.start_vm(class_path=bf.JARS, max_heap_size=max_heap_size)
    global VM_STARTED
    VM_STARTED = True


def JVdone():
    jv.kill_vm()
    global VM_KILLED
    VM_KILLED = True


def parse_xml_metadata_IDS(xml, array_order=DEFAULT_DIM_ORDER):
    root = xml.getroot()
    size_tags = ['Size' + c for c in 'XYZ']
    res_tags = ['PhysicalSize' + c for c in 'XYZ']
    root = xml.getroot()
    for child in root[2].iter():
        if child.tag.endswith('Pixels'):
            pixel_sizes = tuple([float(child.attrib[t])
                                 for t in res_tags])
            pixel_format = tuple([int(child.attrib[t])
                                  for t in size_tags])
            channels = int(child.attrib['SizeC'])
    output={}
    output['format']=pixel_format
    output['pixel']=pixel_sizes
    output['channel']=channels
    return output

def metadata(filename, array_order=DEFAULT_DIM_ORDER):
    md_string = bf.get_omexml_metadata(filename)
    return parse_xml_metadata(md_string, array_order)

run()
