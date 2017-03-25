import cv2
import cv2.cv as cv
from gamera.core import *
from gamera.toolkits.musicstaves import musicstaves_rl_simple
from gamera.toolkits.musicstaves import stafffinder_miyao
import numpy as np
import sys


def detect_staves(img_name):
    init_gamera()

    image = load_image(img_name)
    image = image.to_onebit()
    ancho = image.ncols
    alto = image.nrows

    sf = stafffinder_miyao.StaffFinder_miyao(image)
    sf.find_staves(num_lines=5)
    d = sf.staffspace_height
    n = sf.staffline_height
    staff_segments = []

    staves = sf.get_average()
    for i, staff in enumerate(staves):
        #print "Staff %d has %d staves:" % (i+1, len(staff))
        for j, line in enumerate(staff):
            if j == 0:
                ini = max(line.average_y-2*(d+n), 0)
            if j == len(staff)-1:
                fin = min(line.average_y+2*(d+n), alto-1)
            #print "    %d. line at y-position:" % (j+1), line.average_y
        img2 = SubImage(image, Point(0, ini), Point(ancho-1, fin))
        staff_segments.append(img2)
    return staff_segments, d, n


def detect_noteheads(img_name, min, max):
    img = cv2.imread(img_name, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=min,
                               maxRadius=max)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    staves, space, line = detect_staves("imagenes/"+sys.argv[1])
    staff_segments = []
    for i, staff in enumerate(staves):
        staff_segments.append(staff)
        staff.save_PNG("imagenes/staff_segments/staff{0}.png".format(i+1))

    # LEVEL 0 SEGMENTATION
    level0 = []
    #for s in staff_segments:
    s = staff_segments[0]
    xp = s.projection_cols()
    ini = 0
    in_segment = False
    for i in range(len(xp)):
        if not in_segment and xp[i] > 5*line:
            ini = i
            in_segment = True
        elif in_segment and xp[i] <= 5*line:
            level0.append((ini, i))
            img2 = SubImage(s, Point(ini, s.offset_y), Point(i, s.offset_y+s.nrows-1))
            img2.save_PNG("imagenes/staff_segments/level0/{0}.png".format(len(level0)))
            in_segment = False





    # NOTE HEAD DETECTION



    # LEVEL 1 SEGMENTATION



    # xp = img.projection_cols()
    # yp = img.projection_rows()
