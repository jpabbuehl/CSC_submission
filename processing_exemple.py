'''
Created on Nov 1, 2015

Exemple of lightsheet acquisition
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
from imagescience.shape import Ellipse
from ij.plugin.frame import RoiManager
from math import sqrt, fabs
from java.awt import Color
from ij.plugin.filter import EDM
from java.util import Random
from jarray import zeros
from loci.formats import ImageReader, MetadataTools
from fiji.plugin.trackmate import Model,  Settings, TrackMate, SelectionModel, Logger, Spot
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils, ManualTrackerFactory
from javax.vecmath import Point3f, Tuple3f
from xml.etree import ElementTree as ET
from ij import Prefs
from ij.plugin.filter import Binary
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
from ij.plugin import Duplicator
import java.awt.List
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

Prefs.blackBackground = True


def run():
    imp = IJ.getImage()
    imp.show()
    ch_nucleus = 4
    spot_data = detection(imp, ch_nucleus)
    ES_tolerance = 6.0
    diameter_init = 5.0
    ES_area_max = 150.0
    ES_ctrl_pts = 4
    ES_iteration = 150
    repeat_max = 5
    channel = 5
    segmentation(imp, spot_data, channel, diameter_init, ES_tolerance,
                 ES_area_max, ES_ctrl_pts, ES_iteration, repeat_max)


def detection(imp, c):
    cal = imp.getCalibration()
    model = Model()
    settings = Settings()
    settings.setFrom(imp)
    # Configure detector - Manually determined as best
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = {
        'DO_SUBPIXEL_LOCALIZATION': True,
        'RADIUS': 2.0,
        'TARGET_CHANNEL': c,
        'THRESHOLD': 20.0,
        'DO_MEDIAN_FILTERING': False,
    }
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
        IJ.log("Nothing detected")
        IJ.selectWindow('test')
        IJ.run('Close')
    else:
        selectionModel = SelectionModel(model)
        displayer = HyperStackDisplayer(model, selectionModel, imp)
        displayer.render()
        displayer.refresh()
        # Get spots information
        spots = model.getSpots()
        spotIt = spots.iterator(0, False)
        # Loop through spots and save into files
        # Fetch spot features directly from spot
        sid = []
        x = []
        y = []
        q = []
        r = []
        spotID = 0
        for spot in spotIt:
            spotID = spotID + 1
            sid.append(spotID)
            x.append(spot.getFeature('POSITION_X'))
            y.append(spot.getFeature('POSITION_Y'))
            q.append(spot.getFeature('QUALITY'))
            r.append(spot.getFeature('RADIUS'))
        data = zip(sid, x, y, q, r)
        return data


def segmentation(imp, spot_data, channel, diameter_init, ES_tolerance, ES_area_max, ES_ctrl_pts, ES_iteration, repeat_max):
    # Open files
    cal = imp.getCalibration()
    manager = RoiManager.getInstance()
    if manager is None:
        manager = RoiManager()
    # Prepare log files for output
    options = IS.MEDIAN | IS.AREA | IS.MIN_MAX | IS.CENTROID | IS.PERIMETER | IS.ELLIPSE | IS.SKEWNESS
    convergence = []
    Sintensity = []
    for spot in spot_data:
        repeat = 0
        flag = False
        spotID = int(spot[0])
        Xcenter = (float(spot[1]) / cal.pixelWidth)
        Ycenter = (float(spot[2]) / cal.pixelHeight)
        Quality = float(spot[3])
        diameter_init = float(spot[4] / cal.pixelWidth) * 2.0
        while True:
            manager = RoiManager.getInstance()
            if manager is None:
                manager = RoiManager()
            Xcurrent = int(Xcenter - diameter_init / 2.0)
            Ycurrent = int(Ycenter - diameter_init / 2.0)
            Dcurrent1 = int(diameter_init * (1.2 - repeat / 10.0))
            Dcurrent2 = int(diameter_init * (0.8 + repeat / 10.0))
            roi = OvalRoi(Xcurrent, Ycurrent, Dcurrent1, Dcurrent2)
            imp.setPosition(channel)
            imp.setRoi(roi)
            Esnake_options1 = "target_brightness=Bright control_points=" + \
                str(ES_ctrl_pts) + " gaussian_blur=0 "
            Esnake_options2 = "energy_type=Contour alpha=2.0E-5 max_iterations=" + \
                str(ES_iteration) + " immortal=false"
            IJ.run(imp, "E-Snake", Esnake_options1 + Esnake_options2)
            roi_snake = manager.getRoisAsArray()
            roi_ind = len(roi_snake) - 1
            stats = IS.getStatistics(
                imp.getProcessor(), options, imp.getCalibration())
            perimeter = roi_snake[roi_ind].getLength() * cal.pixelWidth
            circularity = 4.0 * 3.1417 * (stats.area / (perimeter * perimeter))
            if stats.area > 17.0 and stats.area < ES_area_max and stats.skewness < -0.01 and circularity > 0.01 and stats.minor > 2.0 and boundaries(Xcenter, Ycenter, stats.xCentroid / cal.pixelWidth, stats.yCentroid / cal.pixelHeight, ES_tolerance):
                Sintensity = stats.median
                convergence.append(True)
                break
            if stats.median > 6000 and stats.area > 17.0 and stats.area < ES_area_max:
                Sintensity = stats.median
                convergence.append(True)
                break
            elif repeat > repeat_max:
                manager.select(imp, roi_ind)
                manager.runCommand(imp, 'Delete')
                roi = OvalRoi(Xcenter + 1.0 - diameter_init / 2.0, Ycenter +
                              1.0 - diameter_init / 2.0, diameter_init, diameter_init)
                imp.setRoi(roi)
                manager.add(imp, roi, spotID)
                roi_snake.append(roi)
                stats = IS.getStatistics(
                    imp.getProcessor(), options, imp.getCalibration())
                Sintensity = stats.median
                convergence.append(False)
                break
            else:
                IJ.log('Area=' + str(stats.area) + '  Skewness=' + str(stats.skewness) +
                       ' circularity=' + str(circularity) + ' Minor=' + str(stats.minor))
                manager.select(imp, roi_ind)
                manager.runCommand(imp, 'Delete')
                repeat += 1
        # End Spot-segmentation
    # End all Spots-segmentation
    manager.runCommand(imp, 'Show All')
    imp.setPosition(channel)
    color = imp.createImagePlus()
    ip = imp.getProcessor().duplicate()
    color.setProcessor("segmentation" + str(channel), ip)
    color.show()
    IJ.selectWindow("segmentation" + str(channel))
    manager.moveRoisToOverlay(color)
    spot_optimal = manager.getRoisAsArray()
    manager.reset()
    for i in xrange(0, len(spot_optimal)):
        spot = spot_optimal[i]
        spot.setStrokeWidth(2)
        if convergence[i]:
            spot.setStrokeColor(Color.GREEN)
        else:
            spot.setStrokeColor(Color.MAGENTA)
        imp.setRoi(spot)
        manager.add(imp, spot, i)
    manager.runCommand(imp, 'Show All')
    imp.setPosition(channel)


def boundaries(x, y, Xcentroid, Ycentroid, tolerance):
    if (x + tolerance) > Xcentroid and (x - tolerance) < Xcentroid:
        if (y + tolerance) > Ycentroid and (y - tolerance) < Ycentroid:
            return True
    return False

run()
