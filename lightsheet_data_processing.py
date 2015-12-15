'''
Created on Nov 1, 2015

Image processing of Lightsheet acquisition
Paper: Long-term engraftment of primary bone marrow stroma promotes hematopoietic reconstitution after transplantation
Author: Jean-Paul Abbuehl

@author: Jean-Paul Abbuehl
'''
from __future__ import with_statement
from threading import Thread
from ij.io import OpenDialog, DirectoryChooser
from ij import IJ, ImageStack, ImagePlus
from ij.plugin import ChannelSplitter, RGBStackMerge, ImageCalculator
from ij.process import ImageStatistics as IS, FloatProcessor
from ij.measure import Measurements as Measurements
from ij.gui import Roi, PolygonRoi, OvalRoi, GenericDialog, Line
from ij.plugin.frame import RoiManager
from math import sqrt, fabs
from java.util import Random
from java.awt import Color
from jarray import zeros
from loci.formats import ImageReader, MetadataTools
from fiji.plugin.trackmate import Model,  Settings, TrackMate, SelectionModel, Logger, Spot
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils, ManualTrackerFactory
from javax.vecmath import Point3f, Tuple3f
from xml.etree import ElementTree as ET
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.tracking.ManualTrackerFactory as ManualTrackerFactory
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.visualization.SpotColorGenerator as SpotColorGenerator
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
import fiji.plugin.trackmate.detection.DetectorKeys as DetectorKeys
import fiji.plugin.trackmate.features.FeatureAnalyzer as FeatureAnalyzer
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory as SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.action.ExportStatsToIJAction as ExportStatsToIJAction
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.features.SpotFeatureCalculator as SpotFeatureCalculator
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory
import fiji.plugin.trackmate.util.TMUtils as TMUtils
import os
import sys
import re
import csv
import gc
import re
import time
import Queue
import ntpath
import math


def run():
    GUI(True)
    folders()
    channels = 5
	nucleus_channel = 4

	# Convert CZI to IDS, extract data for each view
    CZI_convert(nucleus_channel)

	# Get all files
	AllFiles = get_filepaths(folder1)

	# Spot detector
    for infile in AllFiles:
		# Use 250 for 32 Gb, 350 for 64 Gb, 500 for 128 Gb
		stack_by_stack_process = 350
		animation = True
        nucleus_detection(infile, nucleus_channel,stack_by_stack_process,animation)

	# Correct Z drift
    for infile in AllFiles:
    	Zrepeat=10
    	Zdrift(infile,nucleus_channel,Zrepeat)

	# Snake Segmentation
    for infile in AllFiles:
        Zrepeat = 10
        diameter = 20
        tolerance = 10
        repeat_max = 5
        channel_segmentation(infile, diameter, tolerance, repeat_max, Zrepeat)

	# Shading correction
    for infile in AllFiles1:
		threshold=5.0
		shading_correction(infile, threshold)

	# Generate overview for data analysis
	mip(0.0, 1.0, channels)

def GUI(active):
    # GUI to select file and select input/output folders
    if not active:
        sourcedir = "D:\\JPA\\wt2"
        sourcefile = "D:\\JPA\\wt2.czi"
        targetdir = "D:\\JPA\\wt2"
        XMLinput = "D:\\JPA\\wt2.mvl"
    else:
        od = OpenDialog("Select CZI Zeiss file", None)
        soucedir = od.getDirectory()
        sourcefile = od.getFileName()
        od = OpenDialog("Select MVL Zeiss file", None)
        XMLdir = od.getDirectory()
        XMLfile = od.getFileName()
        XMLinput = os.path.join(XMLdir, XMLfile)
        targetdir = DirectoryChooser(
            "Choose destination folder").getDirectory()
        sourcedir = os.path.join(soucedir, sourcefile)
	# Create some global variables
    global INpath
    INpath = sourcedir
    global source
    source = sourcefile
    global OUTpath
    OUTpath = targetdir
    global XMLzeiss
    XMLzeiss = XMLinput

def folders():
    global folder1
    folder1 = 'view'
    folder1 = os.path.join(OUTpath, folder1)
    global folder2
    folder2 = 'preprocess'
    folder2 = os.path.join(OUTpath, folder2)
    global folder3
    folder3 = 'QC log'
    folder3 = os.path.join(OUTpath, folder3)
    global folder4
    folder4 = 'Zstack log'
    folder4 = os.path.join(OUTpath, folder4)
    global folder5
    folder5 = 'LoG_detector'
    folder5 = os.path.join(OUTpath, folder5)
    global folder6
    folder6 = 'segmentation'
    folder6 = os.path.join(OUTpath, folder6)
    global folder7t
    folder7t = 'Zdrift_temp'
    folder7t = os.path.join(OUTpath, folder7t)
    global folder7
    folder7 = 'Zdrift'
    folder7 = os.path.join(OUTpath, folder7)
    global folder8
    folder8 = 'MIP_temp'
    folder8 = os.path.join(OUTpath, folder8)
    global folder8a
    folder8a = 'MIP'
    folder8a = os.path.join(OUTpath, folder8a)
    global folder9
    folder9 = 'stitched'
    folder9 = os.path.join(OUTpath, folder9)
    global folder10
    folder10 = 'shading'
    folder10 = os.path.join(OUTpath, folder10)
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    if not os.path.exists(folder3):
        os.makedirs(folder3)
    if not os.path.exists(folder4):
        os.makedirs(folder4)
    if not os.path.exists(folder5):
        os.makedirs(folder5)
    if not os.path.exists(folder6):
        os.makedirs(folder6)
    if not os.path.exists(folder7):
        os.makedirs(folder7)
    if not os.path.exists(folder7t):
        os.makedirs(folder7t)
    if not os.path.exists(folder8):
        os.makedirs(folder8)
    if not os.path.exists(folder9):
        os.makedirs(folder9)
    if not os.path.exists(folder10):
        os.makedirs(folder10)


def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]


def percentile(N, percent):
    if not N:
        return None
    N = sorted(N)
    k = float((len(N) - 1.0) * percent)
    f = float(math.floor(k))
    c = float(math.ceil(k))
    return N[int(k)]


def Z1_metadata(sourcefile):
	# Access header of Z1 lighsheet data to determine nb views
    reader = ImageReader()
    omeMeta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(omeMeta)
    reader.setId(sourcefile)
    seriesCount = reader.getSeriesCount()
    reader.close()
    return seriesCount

def get_filepaths(directory):
	# Get fullpath for each file in directory
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
			if infile.endswith(".ids")
            	file_paths.append(filepath)
    return file_paths

def CZI_convert(nucleus_channel):
	# Convert CZI to IDS files
    NbSeries = Z1_metadata(sourcefile)
    IJ.log(INpath + " contains " + str(NbSeries) + " views")
    options = "open=[" + INpath + \
        "] stack_order=XYCZT color_mode=Composite view=Hyperstack"
    for k in range(1, NbSeries + 1):
        imp = IJ.run("Bio-Formats Importer", options + " series_" + str(k))
        output = "view" + str(k) + ".ids"
        IJ.run(imp, "Bio-Formats Exporter", "save=" +
               os.path.join(folder1, output))
		preprocessing(imp,nucleus_channel,output)
        IJ.run("Close")
        IJ.log("View " + str(k) + " Converted")

def filename(fullpath):
	# Extract file name from path:
    head, tail = ntpath.split(fullpath)
    return tail or ntpath.basename(head)

def preprocessing(imp, nucleus_channel,name):
	imp.show()
    name = re.sub('.ids', '', infile)
    # Z Stats of dataset
    Smean = []
    Smax = []
    Sstd = []
    Sstack = []
    for i in xrange(1, imp.getNSlices() + 1):
        imp.setSlice(i)
        options = IS.STD_DEV | IS.MIN_MAX | IS.MEAN
        ip = imp.getProcessor()
        stats = IS.getStatistics(ip, options, imp.getCalibration())
        Sstd.append(stats.stdDev)
        Smax.append(stats.max)
        Smean.append(stats.mean)
        Sstack.append(i)

    # Flag for kicking out view if nothing relevant
    detection_FLAG = False
    Zstart = 1
    Zend = imp.getNSlices() + 1

    # Detect stacks with information
    IJ.run(imp, "Variance...", "radius=5 stack")
    temp = IJ.getImage()
    Tmean = []
    Tstack = []
    for i in xrange(1, temp.getNSlices() + 1):
        temp.setSlice(i)
        options = IS.MEAN
        ip = temp.getProcessor()
        stats = IS.getStatistics(ip, options, temp.getCalibration())
        Tmean.append(stats.mean)
        Tstack.append(i)

    # Half of stacks are useless, so lets kick them out
    stack_threshold = median(Tmean)
    Srelevant = []
    for i in xrange(0, len(Tmean) - 1):
        if(Tmean[i] > stack_threshold * 0.5):
            # Positive staining is always above 1500 during acquisition
            if(Smax[i] > 1500):
                Srelevant.append(True)
            else:
                Srelevant.append(False)

    stack_extra = 10
    if(sum(Srelevant) > 0):
        # Detection of stacks with info
        detection_FLAG = True
        # Add extra stacks to be sure
        for i in xrange(len(Srelevant)):
            if Srelevant[i]:
                Zstart = i - stack_extra
        for i in reversed(xrange(len(Srelevant))):
            if Srelevant[i]:
                Zend = i + stack_extra
        if(Zstart < 1):
            Zstart = 1
        if(Zend > len(Tmean)):
            Zend = len(Tmean)

    output = name + "_channel" + str(nucleus_channel) + "_detection.csv"
    with open(os.path.join(folder3, output), 'wb') as outfile:
        Zwriter = csv.writer(outfile, delimiter=',')
        Sdata = zip(Sstack, Smean, Sstd, Smax, Srelevant)
        Zwriter.writerow(['slice', 'mean', 'std', 'max','relevant'])
        for Srow in Sdata:
            Zwriter.writerow(Srow)

def nucleus_detection(infile, nucleus_channel, stacksize, animation):
	# Detect nucleus with 3d log filters
    fullpath = infile
    infile = filename(infile)
    IJ.log("Start Segmentation " + str(infile))
    # First get Nb Stacks
    reader = ImageReader()
    omeMeta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(omeMeta)
    reader.setId(fullpath)
    default_options = "stack_order=XYCZT color_mode=Composite view=Hyperstack specify_range c_begin=" + \
        str(nucleus_channel) + " c_end=" + str(nucleus_channel) + \
        " c_step=1 open=[" + fullpath + "]"
    NbStack = reader.getSizeZ()
    reader.close()
    output = re.sub('.ids', '.csv', infile)
    with open(os.path.join(folder5, output), 'wb') as outfile:
        DETECTwriter = csv.writer(outfile, delimiter=',')
        DETECTwriter.writerow(
            ['spotID', 'roundID', 'X', 'Y', 'Z', 'QUALITY', 'SNR', 'INTENSITY'])
    rounds = NbStack // stacksize
    spotID = 1
    for roundid in xrange(1, rounds + 2):
        # Process stacksize by stacksize otherwise crash because too many spots
        Zstart = (stacksize * roundid - stacksize + 1)
        Zend = (stacksize * roundid)
        if(Zend > NbStack):
            Zend = NbStack % stacksize + (roundid - 1) * stacksize
        IJ.log("Round:" + str(roundid) + ' Zstart=' + str(Zstart) +
               ' Zend=' + str(Zend) + ' out of ' + str(NbStack))
        IJ.run("Bio-Formats Importer", default_options + " z_begin=" +
               str(Zstart) + " z_end=" + str(Zend) + " z_step=1")
        imp = IJ.getImage()
        imp.show()
        cal = imp.getCalibration()
        model = Model()
        settings = Settings()
        settings.setFrom(imp)
        # Configure detector - Manually determined as best
        settings.detectorFactory = LogDetectorFactory()
        settings.detectorSettings = {
            'DO_SUBPIXEL_LOCALIZATION': True,
            'RADIUS': 5.5,
            'TARGET_CHANNEL': 1,
            'THRESHOLD': 50.0,
            'DO_MEDIAN_FILTERING': False,
        }
        filter1 = FeatureFilter('QUALITY', 1, True)
        settings.addSpotFilter(filter1)
        settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
        settings.trackerFactory = SparseLAPTrackerFactory()
        settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()

        trackmate = TrackMate(model, settings)
        ok = trackmate.checkInput()
        if not ok:
            sys.exit(str(trackmate.getErrorMessage()))
        try:
            ok = trackmate.process()
        except:
            IJ.log("Nothing detected, Round:" + str(roundid) + ' Zstart=' +
                   str(Zstart) + ' Zend=' + str(Zend) + ' out of ' + str(NbStack))
            IJ.selectWindow(infile)
            IJ.run('Close')
            continue
        else:
            if animation:
                # For plotting purpose only
                imp.setPosition(1, 1, imp.getNFrames())
                imp.getProcessor().setMinAndMax(0, 4000)
                selectionModel = SelectionModel(model)
                displayer = HyperStackDisplayer(model, selectionModel, imp)
                displayer.render()
                displayer.refresh()
                for i in xrange(1, imp.getNSlices() + 1):
                    imp.setSlice(i)
                    time.sleep(0.05)
            IJ.selectWindow(infile)
            IJ.run('Close')
            spots = model.getSpots()
            spotIt = spots.iterator(0, False)
            sid = []
            sroundid = []
            x = []
            y = []
            z = []
            q = []
            snr = []
            intensity = []
            for spot in spotIt:
                sid.append(spotID)
                spotID = spotID + 1
                sroundid.append(roundid)
                x.append(spot.getFeature('POSITION_X'))
                y.append(spot.getFeature('POSITION_Y'))
                q.append(spot.getFeature('QUALITY'))
                snr.append(spot.getFeature('SNR'))
                intensity.append(spot.getFeature('MEAN_INTENSITY'))
                # Correct Z position
                correct_z = spot.getFeature(
                    'POSITION_Z') + (roundid - 1) * float(stacksize) * cal.pixelDepth
                z.append(correct_z)
            with open(os.path.join(folder5, output), 'ab') as outfile:
                DETECTwriter = csv.writer(outfile, delimiter=',')
                Sdata = zip(sid, sroundid, x, y, z, q, snr, intensity)
                for Srow in Sdata:
                    DETECTwriter.writerow(Srow)

def Zdrift(infile, nucleus_channel, repeat):
    # Calculate grid-Zshift of channels against nucleus_channel
    default_options = "stack_order=XYCZT color_mode=Grayscale view=Hyperstack"
    IJ.run("Bio-Formats Importer", default_options + " open=[" + infile + "]")
    imp = IJ.getImage()
    short_infile = filename(infile)
    # Calculate Zdrift with X projections, proceed in a grid-fashion
    for i in xrange(1, repeat - 1):
        name = 'view_x' + str(i) + '.txt'
        x_position = i * imp.getWidth() / repeat
        roi = Line(x_position, 1, x_position, imp.getHeight())
        imp.setRoi(roi)
        temp = IJ.run(imp, "Reslice [/]...",
                      "output=0.054 slice_count=1 rotate avoid")
        temp = IJ.run(temp, "Re-order Hyperstack ...",
                      "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
        temp = IJ.run(temp, "Image Stabilizer", "transformation=Translation maximum_pyramid_levels=1 template_update_coefficient=0.90 maximum_iterations=200 error_tolerance=0.0000001 log_transformation_coefficients")
        IJ.selectWindow('Reslice.log')
        IJ.saveAs("Text", os.path.join(folder7t, name))
        IJ.selectWindow(name)
        IJ.run("Close")
        IJ.selectWindow('Reslice of ' + imp.getTitle())
        IJ.run("Close")
    # Calculate Zdrift with Y projections, proceed in a grid-fashion
    ylength = imp.getHeight()
    for i in xrange(1, repeat - 1):
        name = 'view_y' + str(i) + '.txt'
        y_position = i * imp.getHeight() / repeat
        roi = Line(1, y_position, imp.getWidth(), y_position)
        imp.setRoi(roi)
        temp = IJ.run(imp, "Reslice [/]...",
                      "output=0.054 slice_count=1 rotate avoid")
        temp = IJ.run(temp, "Re-order Hyperstack ...",
                      "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
        temp = IJ.run(temp, "Image Stabilizer", "transformation=Translation maximum_pyramid_levels=1 template_update_coefficient=0.90 maximum_iterations=200 error_tolerance=0.0000001 log_transformation_coefficients")
        IJ.selectWindow('Reslice.log')
        IJ.saveAs("Text", os.path.join(folder7t, name))
        IJ.selectWindow(name)
        IJ.run("Close")
        IJ.selectWindow('Reslice of ' + imp.getTitle())
        IJ.run("Close")
    # Make Grid-matrix for Zdrift and return this grid for Z adjustment during color segmentation
    # Beware that channel 1 starts at position 0 and grid position 1 starts at
    # position 0
    IJ.selectWindow(short_infile)
    IJ.run("Close")
    repeat = repeat - 2
    channels = 5
    XZdrift = [[0 for x in range(channels)] for x in range(repeat)]
    YZdrift = XZdrift
    for file in os.listdir(folder7t):
        IJ.log(file)
        with open(os.path.join(folder7t, file), 'rb') as f:
            lines = f.read().splitlines()
            c = []
            for i in xrange(2, channels + 2):
                c.append(float(lines[i].split(',')[2]))
            ground = float(lines[nucleus_channel + 1].split(',')[2])
            index = int(re.findall(r'\d+', file)[0]) - 1
            for i in xrange(0, channels):
                if '_x' in file:
                    XZdrift[index][i] = c[i] - ground
                else:
                    YZdrift[index][i] = c[i] - ground
    # Save XZdrift and YZdrift under file specific system
    output = re.sub('.ids', '.csv', short_infile)
    with open(os.path.join(folder7, 'X-' + output), 'wb') as f:
        DriftLog = csv.writer(f, delimiter=',')
        for i in xrange(0, len(XZdrift)):
            save = [i]
            for c in xrange(0, channels):
                save.append(XZdrift[i][c])
            DriftLog.writerow(save)
    output = re.sub('.ids', '.csv', short_infile)
    with open(os.path.join(folder7, 'Y-' + output), 'wb') as f:
        DriftLog = csv.writer(f, delimiter=',')
        for i in xrange(0, len(YZdrift)):
            save = [i]
            for c in xrange(0, channels):
                save.append(YZdrift[i][c])
            DriftLog.writerow(save)

def retrieve_seeds(infile):
    # Retrieve detected nucleus spots from csv files
    data = {}
    name = re.sub('.ids', '.csv', infile)
    with open(os.path.join(folder5, name), 'rb') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"',
                            skipinitialspace=True)
        header = reader.next()
        for name in header:
            data[name] = []
        for row in reader:
            for i, value in enumerate(row):
                data[header[i]].append(value)
    output = zip(data['spotID'], data['roundID'], data['X'], data['Y'], data[
                 'Z'], data['QUALITY'], data['SNR'], data['INTENSITY'])
    IJ.log('Loading ' + str(len(output)) + ' seed spots from ' + name)
    return output


def retrieve_Zdrift(infile):
    # Retrieve Zdrift for this view
    short_infile = filename(infile)
    output = re.sub('.ids', '.csv', short_infile)
    XZdrift = []
    YZdrift = []
    with open(os.path.join(folder7, 'X-' + output)) as f:
        DriftLog = csv.reader(f, delimiter=',')
        for row in DriftLog:
            input = []
            for i in xrange(1, len(row)):
                input.append(row[i])
            XZdrift.append(input)
    with open(os.path.join(folder7, 'Y-' + output)) as f:
        DriftLog = csv.reader(f, delimiter=',')
        for row in DriftLog:
            input = []
            for i in xrange(1, len(row)):
                input.append(row[i])
            YZdrift.append(input)
    return XZdrift, YZdrift

def boundaries(x, y, Xcentroid, Ycentroid, tolerance):
    # Cell boundaries for convergence of SNAKE
    if (x + tolerance) > Xcentroid and (x - tolerance) < Xcentroid:
        if (y + tolerance) > Ycentroid and (y - tolerance) < Ycentroid:
            return True
    return False

def channel_segmentation(infile, diameter, tolerance, repeat_max, Zrepeat=10):
    # ROI optimization by Esnake optimisation
    default_options = "stack_order=XYCZT color_mode=Grayscale view=Hyperstack"
    IJ.run("Bio-Formats Importer", default_options + " open=[" + infile + "]")
    imp = IJ.getImage()
    cal = imp.getCalibration()
    channels = [i for i in xrange(1, imp.getNChannels() + 1)]

    log = filename(infile)
    log = re.sub('.ids', '.csv', log)
    XZdrift, YZdrift = retrieve_Zdrift(log)
    XZpt = [i * imp.getWidth() / Zrepeat for i in xrange(1, Zrepeat - 1)]
    YZpt = [i * imp.getHeight() / Zrepeat for i in xrange(1, Zrepeat - 1)]

    # Prepare head output file
    for ch in channels:
        csv_name = 'ch' + str(ch) + log
        with open(os.path.join(folder6, csv_name), 'wb') as outfile:
            SegLog = csv.writer(outfile, delimiter=',')
            SegLog.writerow(['spotID', 'Xpos', 'Ypos', 'Zpos',
                             'Quality', 'area', 'intensity', 'min', 'max', 'std'])

    # Retrieve seeds from SpotDetector
    options = IS.MEDIAN | IS.AREA | IS.MIN_MAX | IS.CENTROID
    spots = retrieve_seeds(log)
    for ch in channels:
        for spot in spots:
            repeat = 0
            # Spots positions are given according to calibration, need to
            # convert it to pixel coordinates
            spotID = int(spot[0])
            Xcenter = int(float(spot[2]) / cal.pixelWidth)
            Ycenter = int(float(spot[3]) / cal.pixelHeight)
            Zcenter = float(spot[4]) / cal.pixelDepth
            Quality = float(spot[5])
            # find closest grid location in Zdrift matrix
            Xpt = min(range(len(XZpt)), key=lambda i: abs(XZpt[i] - Xcenter))
            Ypt = min(range(len(YZpt)), key=lambda i: abs(YZpt[i] - Ycenter))
            # Calculate Z position according to SpotZ, calibration and
            # channel-specific Zdrift #
            Zshift = median([float(XZdrift[Xpt][ch - 1]),
                             float(YZdrift[Ypt][ch - 1])]) / cal.pixelDepth
            correctZ = int(Zcenter - Zshift)
            imp.setPosition(ch, correctZ, 1)
            imp.getProcessor().setMinAndMax(0, 3000)
            while True:
                manager = RoiManager.getInstance()
                if manager is None:
                    manager = RoiManager()
                roi = OvalRoi(Xcenter - diameter * (1.0 + repeat / 10.0) / 2.0, Ycenter - diameter * (
                    1.0 + repeat / 10.0) / 2.0, diameter * (1.0 + repeat / 10.0), diameter * (1.0 + repeat / 10.0))
                imp.setRoi(roi)
                IJ.run(imp, "E-Snake", "target_brightness=Bright control_points=3 gaussian_blur=0 energy_type=Mixture alpha=2.0E-5 max_iterations=20 immortal=false")
                roi_snake = manager.getRoisAsArray()[0]
                imp.setRoi(roi_snake)
                stats = IS.getStatistics(
                    imp.getProcessor(), options, imp.getCalibration())
                manager.reset()
                if stats.area > 20.0 and stats.area < 150.0 and boundaries(Xcenter, Ycenter, stats.xCentroid / cal.pixelWidth, stats.yCentroid / cal.pixelHeight, tolerance):
                    Sarea = stats.area
                    Sintensity = stats.median
                    Smin = stats.min
                    Smax = stats.max
                    Sstd = stats.stdDev
                    break
                elif repeat > repeat_max:
                    roi = OvalRoi(Xcenter - diameter / 2.0,
                                  Ycenter - diameter / 2.0, diameter, diameter)
                    imp.setRoi(roi)
                    manager.add(imp, roi, i)
                    stats = IS.getStatistics(
                        imp.getProcessor(), options, imp.getCalibration())
                    Sarea = stats.area
                    Sintensity = stats.median
                    Smin = stats.min
                    Smax = stats.max
                    Sstd = stats.stdDev
                    break
                else:
                    repeat += 1
            # Save results
            csv_name = 'ch' + str(ch) + log
            with open(os.path.join(folder6, csv_name), 'ab') as outfile:
                SegLog = csv.writer(outfile, delimiter=',')
                SegLog.writerow([spotID, Xcenter, Ycenter, correctZ,
                                 Quality, Sarea, Sintensity, Smin, Smax, Sstd])
            # End spot optimization
        # End spots
    # End channels
    IJ.selectWindow(filename(infile))
    IJ.run("Close")

def XMLconfig_process(file):
    # Calculate columns,rows dimensions, extract PixelFormat from MVL Zeiss Files
    tree = ET.parse(file)
    root = tree.getroot()
    data = root[1]

    nb_views = len(data)
    positionX = []
    positionY = []
    PixFormat = int(data[0].attrib['AcquisitionFrameWidth'])
    PixelDepth = float(data[0].attrib['StackStepSize'])
    for i in xrange(0, nb_views):
        positionX.append(float(data[i].attrib['PositionX']))
        positionY.append(float(data[i].attrib['PositionY']))
    positionX = [x - min(positionX) for x in positionX]
    positionY = [y - min(positionY) for y in positionY]
    positionX = [max(positionX) - x for x in positionX]

    Xview_nb = len(list(set(positionX)))
    Yview_nb = len(list(set(positionY)))
    return (Xview_nb, Yview_nb, PixFormat, PixelDepth)

def snake_generator(Ncol, Nrow):
    # Generate two 2d vectors for X,Y according to top left
    # Snake X shift, start top left
    Xsnake = []
    view_id = 1
    for y in xrange(1, Ncol + 2):
        if(y % 2 != 0):
            # Compute from left to right
            Xsnake = Xsnake + ([i for i in xrange(view_id, Ncol + view_id)])
            view_id = max(Xsnake)
        else:
            # Compute from right to left
            Xsnake = Xsnake + \
                ([i for i in xrange(Ncol + view_id, view_id, -1)])
            view_id = max(Xsnake) + 1
        if (view_id > Ncol * Nrow):
            break
    adjust_Xsnake = []
    for y in xrange(0, Ncol + 1):
        start = Ncol * y
        end = start + Ncol
        if end < Ncol * Nrow + 1:
            adjust_Xsnake.append(Xsnake[start:end])
    Xsnake = adjust_Xsnake
    # Snake Y shift, start top right
    Ysnake = []
    ScrollY = []
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
    adjust_Ysnake = []
    for i in xrange(Ncol - 1, -1, -1):
        start = (Nrow) * i
        if(start < 0):
            break
        end = start + Nrow
        adjust_Ysnake.append(Ysnake[start:end])
    Ysnake = adjust_Ysnake
    return (Xsnake, Ysnake)


def shading_correction(infile, threshold):
    # Create artificial shading for stiching collection optimisation
    default_options = "stack_order=XYCZT color_mode=Grayscale view=Hyperstack"
    IJ.run("Bio-Formats Importer", default_options + " open=[" + infile + "]")
    imp = IJ.getImage()
    cal = imp.getCalibration()
    current = ChannelSplitter.split(imp)
    for c in xrange(0, len(current)):
        results = []
        for i in xrange(0, imp.getWidth()):
            roi = Line(0, i, imp.getWidth(), i)
            current[c].show()
            current[c].setRoi(roi)
            temp = IJ.run(current[c], "Reslice [/]...",
                          "output=0.054 slice_count=1 rotate avoid")
            temp = IJ.getImage()
            ip = temp.getProcessor().convertToShort(True)
            pixels = ip.getPixels()
            w = ip.getWidth()
            h = ip.getHeight()
            row = []
            for j in xrange(len(pixels)):
                row.append(pixels[j])
                if j % w == w - 1:
                    results.append(int(percentile(sorted(row), threshold)))
                    row = []
            reslice_names = "Reslice of C" + str(c + 1) + "-" + imp.getTitle()
            reslice_names = re.sub(".ids", "", reslice_names)
            IJ.selectWindow(reslice_names)
            IJ.run("Close")
        imp2 = IJ.createImage("shading_ch" + str(c + 1),
                              "16-bit black", imp.getHeight(), imp.getWidth(), 1)
        pix = imp2.getProcessor().getPixels()
        for i in range(len(pix)):
            pix[i] = results[i]
        imp2.show()
        name = 'ch' + str(c + 1) + imp.getTitle()
        IJ.run(imp2, "Bio-Formats Exporter",
               "save=" + os.path.join(folder10, name))
        IJ.selectWindow("shading_ch" + str(c + 1))
        IJ.run('Close')
        IJ.selectWindow("C" + str(c + 1) + "-" + imp.getTitle())
        IJ.run('Close')


def mip(qmin, qmax, channels):
    files = []
    options = []
    nrow, ncol, PixFormat, PixelDepth = XMLconfig_process(XMLzeiss)
    AllFiles = get_filepaths(folder1)
    for infile in AllFiles:
        if infile.endswith(".ids"):
            files.append(infile)
            if qmin == 0.0 and qmax == 1.0:
                options.append(
                    'stack_order=XYCZT color_mode=Grayscale view=Hyperstack specify_range')
            else:
                Zstart, Zend = selectZrange(
                    infile, float(qmin), float(qmax))
                Zstart = int(Zstart * PixelDepth)
                Zend = int(Zend * PixelDepth)
                options.append('stack_order=XYCZT color_mode=Grayscale view=Hyperstack specify_range z_begin=' + str(
                    Zstart) + ' z_end=' + str(Zend) + ' z_step=1')
    for i in xrange(len(files)):
        for c in xrange(1, channels + 1):
            name = filename(files[i])
            IJ.run("Bio-Formats Importer", options[i] + " open=[" + files[
                   i] + "] c_begin=" + str(c) + " c_end=" + str(c) + " c_step=1")
            imp = IJ.getImage()
            imp.show()
            imp = IJ.run("Z Project...", "projection=[Max Intensity]")
            IJ.selectWindow('MAX_' + name)
            imp = IJ.getImage()
            nb = int(re.search(r'\d+', name).group())
            output = "ch" + str(c) + "view" + str(nb).zfill(2) + ".tif"
            IJ.saveAs(imp, 'tif', os.path.join(folder8, output))
            IJ.run("Close")
            IJ.selectWindow(name)
            IJ.run("Close")
    # Merge all channels into composite image
    files = get_filepaths(folder1)
    for i in xrange(len(files)):
        if files[i].endswith(".ids"):
            name = filename(files[i])
            nb = int(re.search(r'\d+', name).group())
            options = ''
            names = []
            for c in xrange(1, channels + 1):
                current_name = os.path.join(
                    folder8, "ch" + str(c) + "view" + str(nb).zfill(2) + ".tif")
                IJ.log(current_name)
                imp = IJ.openImage(current_name)
                imp.show()
                options = options + " c" + \
                    str(c) + "=ch" + str(c) + "view" + \
                    str(nb).zfill(2) + ".tif"
                names.append("ch" + str(c) + "view" +
                             str(nb).zfill(2) + ".tif")
            IJ.run("Merge Channels...", options + " create")
            IJ.selectWindow("Composite")
            imp = IJ.getImage()
            output = "view" + str(nb).zfill(2) + ".tif"
            IJ.saveAs(imp, 'tif', os.path.join(folder8a, output))
            IJ.run("Close")
    # Now setup Stiching parameters
    default_options1 = "type=[Grid: snake by rows] order=[Right & Down                ] file_names=view{ii}.tif first_file_index_i=1"
    default_options2 = " fusion_method=[Max. Intensity] regression_threshold=0.50 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50"
    default_options3 = " compute_overlap subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)]"
    input = " directory=[" + folder8a + "]"
    output = " output_textfile_name=TileConfiguration.txt output_directory=[" + \
        folder9 + "] image_output=[Write to disk]"
    IJ.run("Grid/Collection stitching", default_options1 + default_options2 + default_options3 +
           input + output + " grid_size_x=" + str(nrow) + " grid_size_y=" + str(ncol) + " tile_overlap=10")
    # Optimize output quality
    files = get_filepaths(folder9)
    for infile in files:
        imp = IJ.openImage(infile)
        imp.show()
        IJ.run(imp, "Enhance Local Contrast (CLAHE)",
               "blocksize=100 histogram=256 maximum=3 mask=*None* fast_(less_accurate)")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")

    # Merge sticthed output into one single RGB image
    options = ''
    for c in xrange(1, channels + 1):
        options = options + " c" + str(c) + "=img_t1_z1_c" + str(c)
    IJ.run("Merge Channels...", options + " create")
    IJ.selectWindow("Composite")
    imp = IJ.getImage()
    imp.setDisplayMode(IJ.COLOR)
    if channels == 4:
        imp.setPosition(1, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run('Cyan')
        imp.setPosition(2, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run('Yellow')
        imp.setPosition(3, 1, 1)
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run('Blue')
        imp.setPosition(4, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run('Red')
    else:
        imp.setPosition(1, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run(imp, 'Cyan')
        imp.setPosition(2, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run(imp, 'Yellow')
        imp.setPosition(3, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run(imp, 'Green')
        imp.setPosition(4, 1, 1)
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run(imp, 'Blue')
        imp.setPosition(5, 1, 1)
        IJ.run(imp, "Subtract...", "value=1500")
        IJ.run(imp, "Enhance Contrast", "saturated=0.2")
        IJ.run(imp, 'Red')

def selectZrange(infile, Qmin, Qmax):
    infile = filename(infile)
    data = retrieve_seeds(infile)
    spotsZ = []
    for row in data:
        spotsZ.append(float(row[4]))
    spotsZ = sorted(spotsZ)
    if len(spotsZ) > 10:
        Zmin = float(percentile(spotsZ, Qmin))
        Zmax = float(percentile(spotsZ, Qmax))
    else:
        Zmin = 1
        Zmax = 3
    IJ.log(infile + ' zmin=' + str(Zmin) + ' zmax=' + str(Zmax))
    return Zmin, Zmax


run()
