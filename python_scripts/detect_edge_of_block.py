# ASSUMPTION: ALL IMAGES ARE CAPTURED FROM ROBOT HAND!!! AT THE SAME POSTURE!!!
import exceptions
import numpy as np
import cv2 as cv
import math
from math import sqrt


def detect_side_of_block(img, testing_param):
    def side_length_seems_fine(min_width_, min_height_, max_width_, max_height_):
        param = 0.25
        length_param = 100
        if abs((max_width_ - min_width_) - (max_height_ - min_height_)) < (max_width_ - min_width_) * param and \
                abs((max_width_ - min_width_) - (max_height_ - min_height_)) < (max_height_ - min_height_) * param \
                and max_width_ - min_width_ > length_param and max_height_ - min_height_ > length_param:
            return True
        else:
            return False

    i = 255
    result_gotten = False
    while i > 5:
        try:
            min_width, min_height, max_width, max_height = \
                detect_side_of_block_main(img, i - 5, i, testing_param)
            if side_length_seems_fine(min_width, min_height, max_width, max_height):
                result_gotten = True
        except exceptions.NotEnoughLinesFoundOnImageError:
            pass
        except exceptions.FindLongestLineError:
            pass
        except exceptions.ParallelLinesExpectedButNoneFoundError:
            pass
        except TypeError:
            pass
        finally:
            i -= 5
            if result_gotten:
                return min_width, min_height, max_width, max_height
    raise exceptions.BoxFindingFailed


def detect_side_of_block_main(img, canny_edge_param_1, canny_edge_param_2, testing_param):
    """takes image, find its face, then find enclosing rectangle - return rectangle dimensions
                                                                   [min_width, min_height, max_width, max_height]"""

    TESTING = testing_param

    def create_blank(__width__, __height__, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        __image__ = np.zeros((__height__, __width__, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        __image__[:] = color

        return __image__

    def calculateDistance(x1_, y1_, x2_, y2_):
        """calculate distance between (x1,y1) and (x2,y2)"""
        dist = math.sqrt((x2_ - x1_) ** 2 + (y2_ - y1_) ** 2)
        return dist

    def find_longest_line(lines__):
        """find longest line in a numpy? array of lines"""
        longest_length = 0
        if len(lines__) == 0:
            raise exceptions.FindLongestLineError("find_longest_line failed, no line is available")
        longest_line = lines__[0]
        for line in lines__:
            for x1, y1, x2, y2 in line:
                length = calculateDistance(x1, y1, x2, y2)
                if length > longest_length:
                    longest_length = length
                    longest_line = line
        return longest_line

    def line_intersect(line1_, line2_):
        line1__ = [[line1_[0], line1_[1]], [line1_[2], line1_[3]]]
        line2__ = [[line2_[0], line2_[1]], [line2_[2], line2_[3]]]
        x_diff = (line1__[0][0] - line1__[1][0], line2__[0][0] - line2__[1][0])
        y_diff = (line1__[0][1] - line1__[1][1], line2__[0][1] - line2__[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(x_diff, y_diff)
        if div == 0:
            raise exceptions.LinesDoNotIntersect

        d = (det(*line1__), det(*line2__))
        x = det(d, x_diff) / div
        y = det(d, y_diff) / div
        return x, y

    def slope(line__):
        [x1_, y1_, x2_, y2_] = line__
        if x2_ - x1_ == 0:
            return 100000
        return (y2_ - y1_) / (x2_ - x1_)

    def length_of_line(line__):
        [x1_, y1_, x2_, y2_] = line__
        return math.sqrt((x2_ - x1_) * (x2_ - x1_) + (y2_ - y1_) * (y2_ - y1_))

    def test_if_parallel(line1_, line2_):
        # return True if close enough to count as parallel, return false if not close enough
        parameter = 0.2
        [x1_, y1_, x2_, y2_] = line1_
        [x1__, y1__, x2__, y2__] = line2_
        return (minDistanceLineSegmentAndPoint([x1__ + x2_ - x1_, y1__ + y2_ - y1_], line2_) / length_of_line(line1_) < parameter) or \
               (minDistanceLineSegmentAndPoint([x1__ - x2_ + x1_, y1__ - y2_ + y1_], line2_) / length_of_line(line1_) < parameter) or \
               (minDistanceLineSegmentAndPoint([x1_ + x2__ - x1__, y1_ + y2__ - y1__], line1_) / length_of_line(
                   line2_) < parameter) or \
               (minDistanceLineSegmentAndPoint([x1_ - x2__ + x1__, y1_ - y2__ + y1__], line1_) / length_of_line(line2_) < parameter)

    def test_if_perpendicular(line1_, line2_):
        # return True if close enough to count as perpendicular, return false if not close enough
        parameter = 0.88
        [x1_, y1_, x2_, y2_] = line1_
        [x1__, y1__, x2__, y2__] = line2_
        return (minDistanceLineSegmentAndPoint([x1__ + x2_ - x1_, y1__ + y2_ - y1_], line2_) / length_of_line(
            line1_) > parameter) and \
               (minDistanceLineSegmentAndPoint([x1__ - x2_ + x1_, y1__ - y2_ + y1_], line2_) / length_of_line(
                   line1_) > parameter) and \
               (minDistanceLineSegmentAndPoint([x1_ + x2__ - x1__, y1_ + y2__ - y1__], line1_) / length_of_line(
                   line2_) > parameter) and \
               (minDistanceLineSegmentAndPoint([x1_ - x2__ + x1__, y1_ - y2__ + y1__], line1_) / length_of_line(line2_) > parameter)

    def minDistanceLineSegmentAndPoint(E, line_AB):

        A = [line_AB[0], line_AB[1]]
        B = [line_AB[2], line_AB[3]]

        # vector AB
        AB = [None, None]
        AB[0] = B[0] - A[0]
        AB[1] = B[1] - A[1]

        # vector BP
        BE = [None, None]
        BE[0] = E[0] - B[0]
        BE[1] = E[1] - B[1]

        # vector AP
        AE = [None, None]
        AE[0] = E[0] - A[0]
        AE[1] = E[1] - A[1]

        # Variables to store dot product

        # Calculating the dot product
        AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
        AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

        # Minimum distance from
        # point E to the line segment
        reqAns = 0

        # Case 1
        if AB_BE > 0:

            # Finding the magnitude
            y = E[1] - B[1]
            x = E[0] - B[0]
            reqAns = sqrt(x * x + y * y)

        # Case 2
        elif AB_AE < 0:
            y = E[1] - A[1]
            x = E[0] - A[0]
            reqAns = sqrt(x * x + y * y)

        # Case 3
        else:

            # Finding the perpendicular distance
            x1 = AB[0]
            y1 = AB[1]
            x2 = AE[0]
            y2 = AE[1]
            mod = sqrt(x1 * x1 + y1 * y1)
            reqAns = abs(x1 * y2 - y1 * x2) / mod

        return reqAns

    def test_if_lines_too_close(line1_, line2_):

        # Function to find distance

        similarity_ = 0
        testing_parameter_ = 40
        if minDistanceLineSegmentAndPoint([line2_[0], line2_[1]], line1_) < testing_parameter_:
            similarity_ += 1
        if minDistanceLineSegmentAndPoint([line2_[2], line2_[3]], line1_) < testing_parameter_:
            similarity_ += 1
        if minDistanceLineSegmentAndPoint([line1_[0], line1_[1]], line2_) < testing_parameter_:
            similarity_ += 1
        if minDistanceLineSegmentAndPoint([line1_[2], line1_[3]], line2_) < testing_parameter_:
            similarity_ += 1
        if similarity_ >= 2:
            return True
        else:
            return False

    def find_longest_parallel_lines(lines__):
        try:
            lines_ = lines__
            found = False
            lines_to_be_tested = []
            while not found:
                next_longest_line = find_longest_line(lines_)
                temp_lists_ = lines_.tolist()
                temp_lists_.remove(next_longest_line.tolist())
                lines_ = np.array(temp_lists_)
                for line__ in lines_to_be_tested:
                    if not found and test_if_parallel(line__[0], next_longest_line[0]) \
                            and not test_if_lines_too_close(line__[0], next_longest_line[0]):
                        return line__, next_longest_line
                lines_to_be_tested.append(next_longest_line)
        except IndexError:
            raise exceptions.ParallelLinesExpectedButNoneFoundError
        except exceptions.FindLongestLineError:
            raise exceptions.ParallelLinesExpectedButNoneFoundError

    # The main function that returns true if
    # the line segment 'p1q1' and 'p2q2' intersect.
    def doIntersect(line1_, line2_):
        # Given three collinear points p, q, r, the function checks if
        # point q lies on line segment 'pr'
        def onSegment(p, q, r):
            if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
                    (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
                return True
            return False

        def orientation(p, q, r):
            # to find the orientation of an ordered triplet (p,q,r)
            # function returns the following values:
            # 0 : Collinear points
            # 1 : Clockwise points
            # 2 : Counterclockwise

            # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
            # for details of below formula.

            val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
            if val > 0:

                # Clockwise orientation
                return 1
            elif val < 0:

                # Counterclockwise orientation
                return 2
            else:

                # Collinear orientation
                return 0

        p1_ = [line1_[0], line1_[1]]
        q1_ = [line1_[2], line1_[3]]
        p2_ = [line2_[0], line2_[1]]
        q2_ = [line2_[2], line2_[3]]

        # Find the 4 orientations required for
        # the general and special cases
        o1 = orientation(p1_, q1_, p2_)
        o2 = orientation(p1_, q1_, q2_)
        o3 = orientation(p2_, q2_, p1_)
        o4 = orientation(p2_, q2_, q1_)

        # General case
        if (o1 != o2) and (o3 != o4):
            return True

        # Special Cases

        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if (o1 == 0) and onSegment(p1_, p2_, q1_):
            return True

        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if (o2 == 0) and onSegment(p1_, q2_, q1_):
            return True

        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if (o3 == 0) and onSegment(p2_, p1_, q2_):
            return True

        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if (o4 == 0) and onSegment(p2_, q1_, q2_):
            return True

        # If none of the cases
        return False

    def lines_intersects_at_at_least_one_corner(line1_, line2_, line3_, line4_):
        param = 5

        return \
            (doIntersect(line1_, line3_) or doIntersect(line1_, line4_) or
             minDistanceLineSegmentAndPoint([line1_[0], line1_[1]], line3_) < param or
             minDistanceLineSegmentAndPoint([line1_[2], line1_[3]], line3_) < param or
             minDistanceLineSegmentAndPoint([line1_[0], line1_[1]], line4_) < param or
             minDistanceLineSegmentAndPoint([line1_[2], line1_[3]], line4_) < param) and \
            (doIntersect(line2_, line3_) or doIntersect(line2_, line4_) or
             minDistanceLineSegmentAndPoint([line2_[0], line2_[1]], line3_) < param or
             minDistanceLineSegmentAndPoint([line2_[2], line2_[3]], line3_) < param or
             minDistanceLineSegmentAndPoint([line2_[0], line2_[1]], line4_) < param or
             minDistanceLineSegmentAndPoint([line2_[2], line2_[3]], line4_) < param) and \
            (doIntersect(line3_, line1_) or doIntersect(line3_, line2_) or
             minDistanceLineSegmentAndPoint([line3_[0], line3_[1]], line1_) < param or
             minDistanceLineSegmentAndPoint([line3_[2], line3_[3]], line1_) < param or
             minDistanceLineSegmentAndPoint([line3_[0], line3_[1]], line2_) < param or
             minDistanceLineSegmentAndPoint([line3_[2], line3_[3]], line2_) < param) and \
            (doIntersect(line4_, line1_) or doIntersect(line4_, line2_) or
             minDistanceLineSegmentAndPoint([line4_[0], line4_[1]], line1_) < param or
             minDistanceLineSegmentAndPoint([line4_[2], line4_[3]], line1_) < param or
             minDistanceLineSegmentAndPoint([line4_[0], line4_[1]], line2_) < param or
             minDistanceLineSegmentAndPoint([line4_[2], line4_[3]], line2_) < param)

    def side_length_seems_fine(line1_, line2_, line3_, line4_):
        param1 = 100
        param2 = 22  # TODO - Automate this
        try:
            return \
                length_of_line([line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1],
                                line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1]]) > param1 and \
                length_of_line([line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1],
                                line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1]]) > param1 and \
                length_of_line([line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1],
                                line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1]]) > param1 and \
                length_of_line([line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1],
                                line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1]]) > param1 and \
                abs(length_of_line([line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1],
                                    line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1]]) -
                    length_of_line([line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1],
                                    line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1]])) < param2 and \
                abs(length_of_line([line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1],
                                    line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1]]) -
                    length_of_line([line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1],
                                    line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1]])) < param2 and \
                abs(length_of_line([line_intersect(line2_, line4_)[0], line_intersect(line2_, line4_)[1],
                                    line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1]]) -
                    length_of_line([line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1],
                                    line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1]])) < param2 and \
                abs(length_of_line([line_intersect(line4_, line1_)[0], line_intersect(line4_, line1_)[1],
                                    line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1]]) -
                    length_of_line([line_intersect(line1_, line3_)[0], line_intersect(line1_, line3_)[1],
                                    line_intersect(line3_, line2_)[0], line_intersect(line3_, line2_)[1]])) < param2

        except exceptions.LinesDoNotIntersect:
            return False

    blurred_img = cv.GaussianBlur(img, (7, 7), 1)
    grayed_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)

    # Canny edge detector
    canny_img = cv.Canny(grayed_img, canny_edge_param_1, canny_edge_param_2)

    kernel = np.ones((2, 2))
    dil_img = cv.dilate(canny_img, kernel, iterations=1)

    if TESTING:
        cv.imshow('Result', dil_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 8  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(dil_img, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)

    if TESTING:
        img_to_draw_the_contour_lines_on3 = img.copy()
        for line in lines:
            [x1, y1, x2, y2] = line[0]
            cv.line(img_to_draw_the_contour_lines_on3, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv.imshow('Result', img_to_draw_the_contour_lines_on3)
        cv.waitKey(0)

    # part 3

    parallel_lines = []
    n = len(lines)
    for i in range(0, n-1):
        for j in range(i+1, n-1):
            if test_if_parallel(lines[i][0], lines[j][0]):
                parallel_lines.append([lines[i], lines[j]])

    not_found = True
    n = len(parallel_lines)
    for i in range(0, n-1):
        for j in range(i+1, n-1):
            if not_found:
                [line1, line2] = parallel_lines[i]
                [line3, line4] = parallel_lines[j]
                if test_if_perpendicular(line1[0], line3[0]) and test_if_perpendicular(line2[0], line3[0]) and \
                        test_if_perpendicular(line1[0], line4[0]) and test_if_perpendicular(line2[0], line4[0]) and \
                        lines_intersects_at_at_least_one_corner(line1[0], line2[0], line3[0], line4[0]) and \
                        side_length_seems_fine(line1[0], line2[0], line3[0], line4[0]):
                    not_found = False

    if not_found:
        raise exceptions.NotEnoughLinesFoundOnImageError
    else:
        final_lines = [line1, line2, line3, line4]

    if TESTING:
        img_to_draw_the_contour_lines_on2 = img.copy()
        for line_ in final_lines:
            [x1, y1, x2, y2] = line_[0]
            cv.line(img_to_draw_the_contour_lines_on2, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv.imshow('lines', img_to_draw_the_contour_lines_on2)
        cv.waitKey(0)

    # part 5
    # next, find smallest rectangle that encloses the region of interest
    # from intersections of the lines
    if TESTING:
        print(final_lines)
    intersect1 = line_intersect(final_lines[0][0], final_lines[2][0])
    intersect2 = line_intersect(final_lines[0][0], final_lines[3][0])
    intersect3 = line_intersect(final_lines[1][0], final_lines[2][0])
    intersect4 = line_intersect(final_lines[1][0], final_lines[3][0])
    intersects = [intersect1, intersect2, intersect3, intersect4]
    max_height = int(max([intersect[1] for intersect in intersects]))
    min_height = int(min([intersect[1] for intersect in intersects]))
    max_width = int(max([intersect[0] for intersect in intersects]))
    min_width = int(min([intersect[0] for intersect in intersects]))

    return min_width, min_height, max_width, max_height
