import cv2

from .post_processing import PostProcessor

aruco = cv2.aruco

postprocessor = PostProcessor()

# @dataclass
class ArucoDetector():

    def __init__(self):
        pass

    def detect(self, imglst, dictionary, parameters):

        d_corners = []
        d_ids = []
        for idx, path in enumerate(imglst):
            # print(colored(os.path.basename(path), 'green'))
            img = cv2.imread(path)
            if idx == 0:
                baseimg = img

            corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary, parameters=parameters)

            if len(corners) == 0:
                print("*** mark is not found ***")
                continue
            else:
                d_ids.extend(ids)

                if idx == 0:
                    d_corners.extend(corners)
                else:
                    M = postprocessor.calc_homography_matrix(baseimg, img)
                    rotated = transform_cood(corners, M)
                    d_corners.extend(rotated)

        return d_corners, d_ids


