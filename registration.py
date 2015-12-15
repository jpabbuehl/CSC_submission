'''
Created on Nov 2, 2015

Iterative closest point algorithm, with snake-tiling acquisition
use nuclear spots as landmark correspondance
Paper: Long-term engraftment of primary bone marrow stroma promotes hematopoietic reconstitution after transplantation
Author: Jean-Paul Abbuehl

@author: Jean-Paul Abbuehl
'''
from scipy.spatial import distance
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
from mpl_toolkits.mplot3d import axes3d, Axes3D


class Point3f(object):

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vec = np.array([x, y, z])

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

    def distance(self, target):
        return np.linalg.norm(self.vec - target.vec)


class Point(object):

    def __init__(self, x, y, z, i):
        self.p = Point3f(x, y, z)
        self.id = i
        self.distances = {}
        self.targets = []

    def toVector(self):
        return [self.p.getX(), self.p.getY(), self.p.getZ()]

    def spotID(self):
        return self.id

    def addtarget(self, point):
        self.targets.append(point.id)
        self.distances[point] = self.p.distance(point.p)

    def addtargets(self, cloud):
        for point in cloud.points:
            self.addtarget(point)

    def findclosest(self):
        dist = 1000000.0  # Out of scope on purpose
        flag = False
        for p, d in self.distances.iteritems():
            if d < dist:
                closest = p
                dist = d
                flag = True
        if flag:
            return closest, dist
        else:
            print 'error'

    def translation(self, point):
        return [x1 - x2 for (x1, x2) in zip(self.toVector(), point.toVector())]

    def mse(self, point):
        diff = self.translation(point)
        return reduce(lambda x, y: x + y * y, diff) / float(len(diff))


class Cloud(object):

    def __init__(self):
        self.points = []
        self.error = []

    def addpoint(self, point):
        self.points.append(point)

    def fitcloud(self, cloud):
        self.error = []
        TFx = []
        TFy = []
        TFz = []
        vector = np.zeros(3)
        for point in self.points:
            point.addtargets(cloud)
            closest, distance = point.findclosest()
            TFvector = point.translation(closest)
            TFx.append(TFvector[0])
            TFy.append(TFvector[1])
            TFz.append(TFvector[2])
            self.error.append(point.mse(closest))
        # calculate average vector transformation
        temp = np.array(np.column_stack((TFx, TFy, TFz)))
        vector = -np.median(temp, axis=0)
        return vector

    def getPoints(self):
        output = []
        for point in self.points:
            output.append(point.toVector())
        return output

    def mse(self):
        return reduce(lambda x, y: x + y, self.error)

    def transform(self, vector3d):
        self.error = []
        adjusted_cloud = Cloud()
        for i in xrange(0, len(self.points)):
            Adj_pt = [x1 + x2 for (x1, x2)
                      in zip(self.points[i].toVector(), vector3d)]
            adjusted_cloud.addpoint(Point(Adj_pt[0], Adj_pt[1], Adj_pt[2], i))
        return adjusted_cloud

    def match(self, cloud, max_dist):
        source_list = []
        match_list = []
        for point in self.points:
            point.addtargets(cloud)
            target, distance = point.findclosest()
            if distance < max_dist:
                source_list.append(point.id)
                match_list.append(target.id)
        return np.column_stack((np.array(source_list), np.array(match_list)))


def icp(overlap, niter, plot):
    Ncol, Nrow, totalPix = XMLconfig_process(XMLzeiss)
    Xsnake, Ysnake = snake_generator(Ncol, Nrow)
    OverlapPix = float(totalPix) * float(overlap / 100.0)
    Xtrans_all = []
    Ytrans_all = []
    # 50 highest spot quality only
    ntop = 50
    # By 1st to last row, from left to right
    for row in Xsnake:
        Xtrans = []
        for i in xrange(0, len(row) - 1):
            # Horizontal processing
            view1 = row[i]
            view2 = row[i + 1]
            data1 = retrieve_seeds('view' + str(view1) + '.csv')
            data2 = retrieve_seeds('view' + str(view2) + '.csv')
            # Filter spots for those in overlapping regions
            xlimit1 = (float(totalPix) - OverlapPix) * pixelSize
            spots1 = data1[data1[:, 1] > xlimit1, :]
            spots1 = spots1[spots1[:, 1] < float(
                totalPix) * pixelSize - 1.0, :]
            # Filter for highest quality lOG
            arr1 = spots1[:, 4]
            Index1 = arr1.argsort()[-ntop:][::-1]
            spots1 = spots1[Index1, :]
            # Adjust spots1 X position for comparison
            spots1[:, 1] = spots1[:, 1] - xlimit1
            # Filter spots for those in overlapping regions
            xlimit2 = OverlapPix * pixelSize
            spots2 = data2[data2[:, 1] < xlimit2, :]
            spots2 = spots2[spots2[:, 1] > 0.0, :]
            # Filter for highest quality lOG
            arr2 = spots2[:, 4]
            Index2 = arr2.argsort()[-ntop:][::-1]
            spots2 = spots2[Index2, :]
            # Matches cloud to find translation vector
            if len(spots1) > 0 and len(spots2) > 0:
                Xtrans_vector, spotID_match = cloud_translation(
                    spots1, spots2, niter, plot)
                Xtrans.append(Xtrans_vector)
                duplicated_spots_between_view(
                    view1, view2, 'horizontal', spotID_match)
            else:
                Xtrans.append([0, 0, 0])
        Xtrans_all.append(Xtrans)
    for row in Ysnake:
        Ytrans = []
        for i in xrange(0, len(row) - 1):
            # Vertical processing
            view1 = row[i]
            view2 = row[i + 1]
            data1 = retrieve_seeds('view' + str(view1) + '.csv')
            data2 = retrieve_seeds('view' + str(view2) + '.csv')
            # Filter spots for those in overlapping regions
            ylimit1 = (float(totalPix) - OverlapPix) * pixelSize
            spots1 = data1[data1[:, 2] > ylimit1, :]
            spots1 = spots1[spots1[:, 2] < float(
                totalPix) * pixelSize - 1.0, :]
            # Filter for highest quality lOG
            arr1 = spots1[:, 4]
            Index1 = arr1.argsort()[-ntop:][::-1]
            spots1 = spots1[Index1, :]
            # Adjust spots1 X position for comparison
            spots1[:, 2] = spots1[:, 2] - ylimit1
            # Filter spots for those in overlapping regions
            ylimit2 = OverlapPix * pixelSize
            spots2 = data2[data2[:, 2] < ylimit2, :]
            spots2 = spots2[spots2[:, 2] > 0.0, :]
            # Filter for highest quality lOG
            arr2 = spots2[:, 4]
            Index2 = arr2.argsort()[-ntop:][::-1]
            spots2 = spots2[Index2, :]
            # Matches cloud to find translation vector
            if len(spots1) > 0 and len(spots2) > 0:
                Ytrans_vector, spotID_match = cloud_translation(
                    spots1, spots2, niter, plot)
                Ytrans.append(Ytrans_vector)
                duplicated_spots_between_view(
                    view1, view2, 'vertical', spotID_match)
            else:
                Ytrans.append([0, 0, 0])
        Ytrans_all.append(Ytrans)
    # Xregistration output

    np.save('Ytrans_all.npy', Ytrans_all)
    np.save('Xtrans_all.npy', Xtrans_all)
    np.save('Xsnake.npy', Xsnake)
    np.save('Ysnake.npy', Ysnake)
    Xregistration = np.zeros((len(Xsnake) - 1, len(Ysnake) - 1))
    print Xregistration
    npxsnake = np.array(Xsnake)
    print npxsnake.shape
    for y in xrange(0, npxsnake.shape(0) - 1):
        current_sum = 0.0
        for x in xrange(0, npxsnake.shape(1) - 1):
            view1 = npxsnake[y, x]
            view2 = npxsnake[y, x + 1]
            current_sum += Xtrans_all[x][y][0]
    # Z translation vector
    if Ncol > Nrow:
        diag = Nrow - 1
    else:
        diag = Ncol - 1
    Zregistration = [0.0] * Ncol * Nrow
    for d in xrange(0, diag):
        print Xsnake[d]
        print Ysnake[d]
        for y in xrange(1, Nrow):
            print 'y'
            print y
            print 'id ysnake'
            print Ysnake[d][y - 1]
            print d
            print Xtrans_all[y - 1]
            current_sum = (Zregistration[Xsnake[
                           y - 1][d]] + Zregistration[Ysnake[d][y - 1]]) / 2.0 + Xtrans_all[y - 1][d][2]
            Zregistration[Xsnake[y][d]] = current_sum
        for x in xrange(1, Ncol):
            print 'x'
            print x
            print Xsnake[d][x - 1]
            current_sum = (Zregistration[Xsnake[d][
                           x - 1]] + Zregistration[Ysnake[d][x - 1]]) / 2.0 + Ytrans_all[x - 1][d][2]
            Zregistration[Ysnake[x][d]] = current_sum
        print Zregistration

    viewID = [i for i in xrange(1, Ncol * Nrow + 1)]
    sData = zip(viewID, Xregistration, Yregistration, Zregistration)
    with open('Registration_log.csv', 'wb') as outfile:
        RegLog = csv.writer(outfile, delimiter=',')
        RegLog.writerow(['IDview', 'Xshift', 'Yshift', 'Zshift'])
        for row in sData:
            RegLog.writerow([row[0], row[1], row[2], row[3]])


def normalizeNP(list1, list2):
    dim1 = list1.shape
    dim2 = list2.shape
    together = np.concatenate((list1[:, 1:], list2[:, 1:]))
    norm_together = prenormalize(together, axis=0)
    list1 = np.hstack((list1[:, 0:1], norm_together[0:dim1[0], :]))
    list2 = np.hstack((list2[:, 0:1], norm_together[dim1[0]:, :]))
    return list1, list2


def getCentroid(list1, list2):
    centroid1 = np.median(list1[:, 1:4], axis=0) / 2.0
    centroid2 = np.median(list2[:, 1:4], axis=0) / 2.0
    return (centroid2 - centroid1)
# Fit cloud1 to cloud2 by translation


def cloud_translation(list1, list2, niter, plot):
    # Prepare cloud
    cloud1 = Cloud()
    cloud2 = Cloud()
    #list1, list2 = normalizeNP(list1,list2)
    for pt1 in list1:
        pt1 = np.ravel(pt1)
        cloud1.addpoint(Point(pt1[1], pt1[2], pt1[3], pt1[0]))
    for pt2 in list2:
        pt2 = np.ravel(pt2)
        cloud2.addpoint(Point(pt2[1], pt2[2], pt2[3], pt2[0]))
    # Cloud fitting
    if plot == '3d':
        cloud_3dplotting(cloud1, cloud2)
    if plot == '2d':
        cloud_2dplotting(cloud1, cloud2, title='before translation')
    final_vector = getCentroid(list1, list2)
    cloud1 = cloud1.transform(final_vector)
    if plot == '3d':
        cloud_3dplotting(cloud1, cloud2)
    if plot == '2d':
        cloud_2dplotting(cloud1, cloud2, title='after translation')
    print final_vector
    iteration = 1
    while True:
        translation_vector = cloud1.fitcloud(cloud2)
        mse = cloud1.mse()
        cloud1 = cloud1.transform(translation_vector)
        final_vector = [
            x1 + x2 for (x1, x2) in zip(final_vector, translation_vector)]
        if plot == '3d':
            cloud_3dplotting(cloud1, cloud2)
        if plot == '2d':
            cloud_2dplotting(
                cloud1, cloud2, title='optimization ' + str(iteration))
        if mse < 1.0 or iteration > niter:
            break
        else:
            iteration += 1
            print 'Iteration ' + str(iteration)
            print 'Final vector  [' + str(final_vector[0]) + ',' + str(final_vector[1]) + ',' + str(final_vector[2]) + ']'
            print 'MSE ' + str(mse)
    spotsID = cloud1.match(cloud2, 500.0)
    spotsID.astype(int)
    return final_vector, spotsID

# Write log file with spot detected in two adjacent views


def cloud_3dplotting(cloud1, cloud2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot points in 3D
    class1 = np.array(cloud1.getPoints()) / pixelSize
    class2 = np.array(cloud2.getPoints()) / pixelSize
    class1 = class1.astype(int)
    class2 = class2.astype(int)
    for i in xrange(class1.shape[0]):
        ax.scatter(class1[i][0], class1[i][1], class1[i][2], c='r', marker='o')
    for i in xrange(class2.shape[0]):
        ax.scatter(class2[i][0], class2[i][1], class2[i][2], c='b', marker='^')
    # Formating graph
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def cloud_2dplotting(cloud1, cloud2, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    # plot points in 2D
    class1 = np.array(cloud1.getPoints()) / pixelSize
    class2 = np.array(cloud2.getPoints()) / pixelSize
    class1 = class1.astype(int)
    class2 = class2.astype(int)
    for i in xrange(class1.shape[0]):
        scatter(class1[i][0], class1[i][1], c='r', marker='o', s=100)
    for i in xrange(class2.shape[0]):
        scatter(class2[i][0], class2[i][1], c='b', marker='^', s=100)
    ax.text(2, 6, title, fontsize=15)
    # Formating graph
    plt.show()


def data_plotting(data1, data2):
    plt.figure()
    # plot points in 2D
    class1 = data1.astype(int)
    class2 = data2.astype(int)
    for i in xrange(class1.shape[0]):
        scatter(class1[i, 0], class1[i, 1], c='r', marker='o')
    for i in xrange(class2.shape[0]):
        scatter(class2[i, 0], class2[i, 1], c='b', marker='^')
    # Formating graph
    plt.show()


def retrieve_seeds(infile):
    name = re.sub('.ids', '.csv', infile)
    with open(os.path.join(folder5, name), 'rb') as f:
        reader = pandas.read_csv(f, sep=',')
        data = reader.values
        data = np.array(data)
        data = data[:, [0, 2, 3, 4, 5]]
        return data
