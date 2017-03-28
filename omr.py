import cv2
from gamera.core import *
from gamera.toolkits.musicstaves import musicstaves_rl_simple
from gamera.toolkits.musicstaves import stafffinder_miyao
import numpy as np
import sys


def detect_staves(img_name):
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


def remove_staves(img_name):
    image = load_image(img_name)
    image = image.to_onebit()
    ms = musicstaves_rl_simple.MusicStaves_rl_simple(image)
    ms.remove_staves(num_lines=5)
    return ms.image


def detect_noteheads(img_name, min, max):
    img = cv2.imread(img_name, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1.5, 10, maxRadius=max)
    print circles

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def rle(array):
#     for i in range(len(array)):


def get_row(img, i):
    return [img.get((j, i)) for j in range(img.ncols)]


def get_col(img, j):
    return [img.get((j, i)) for i in range(img.nrows)]


def rle(array):
    run_length = 0
    run_start = 0
    current_value = 0  # empieza a contar los blancos
    r = []
    for i in range(len(array)):
        if current_value == array[i]:
            run_length += 1
        else:
            r.append((run_start, run_length))
            run_length = 1
            run_start = i
            if current_value:
                current_value = 0
            else:
                current_value = 1
    r.append((run_start, run_length))
    return r


if __name__ == '__main__':
    init_gamera()
    staves, space, line = detect_staves("imagenes/"+sys.argv[1])
    staff_segments = []
    for i, staff in enumerate(staves):
        staff_segments.append(staff)
        staff.save_PNG("imagenes/staff_segments/staff{0}.png".format(i+1))

    # LEVEL 0 SEGMENTATION
    level0 = []
    contador = 0
    for s in staff_segments:
        s = staff_segments[0]
        xp = s.projection_cols()
        ini = 0
        in_segment = False
        for i in range(len(xp)):
            if not in_segment and xp[i] > 5*line:
                ini = i
                in_segment = True
            elif in_segment and xp[i] <= 5*line:
                contador += 1

                filename = "imagenes/staff_segments/level0/{0}.png".format(contador)
                img = SubImage(s, Point(ini-2, s.offset_y), Point(i, s.offset_y+s.nrows-1))
                img.save_PNG(filename)

                img = remove_staves(filename)
                img = img.trim_image()
                img.save_PNG(filename)

                ccs = img.cc_analysis()
                for c, cc in enumerate(ccs):
                    filename = "imagenes/staff_segments/level0/ccs/{0}.png".format((contador, c))
                    cc.save_PNG(filename)
                    img = load_image(filename)
                    level0.append(img)
                in_segment = False

    # NOTE HEAD DETECTION
    for img in level0:
        for col in range(img.ncols):
            black_runs = rle(get_col(img, col))[1::2]
            for run in black_runs:
                if run[1] >= 2*space:
                    for row in range(run[0]+run[1]/2, run[0]+run[1]/2+space/4):
                        img.set(Point(col, row), 0)
        ccs = img.cc_analysis()
        for c, cc in enumerate(ccs):
            yp = cc.projection_rows()
            for i in range(len(yp)):
                if yp[i] < space/2:
                    for j in range(cc.ncols):
                        cc.set(Point(j, i), 0)
            img2 = cc.trim_image()
            img2.save_PNG('red_neuronal/input/trim{0}.png'.format(c))
    #img.save_PNG('img.png')


    # LEVEL 1 SEGMENTATION
