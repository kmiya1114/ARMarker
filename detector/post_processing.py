import cv2
import time



class PostProcessor():

    def __init__(self):
        pass

    # Calculate homography matrix with feature points
    def calc_homography_matrix(self, baseimg, img):
        
        detector = cv2.AKAZE_create()
        # detector = cv2.ORB_create(500)
        b = time.time()
        kp_base, des_base = detector.detectAndCompute(baseimg, None)
        kp_tag, des_tag = detector.detectAndCompute(img, None)
        print("detect keypoint：", time.time() - b)

        ### flann ###
        c = time.time()
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)

        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_base, des_tag, k=2)

        ### bf ###
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des_base, des_tag, k=2)

        ratio = 0.1
        good = [[m] for m, n in matches if m.distance < ratio * n.distance]
        print("good match：", time.time() - c)

        if len(good) > 10:
            base_pt = np.float32([kp_base[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            tag_pt = np.float32([kp_tag[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(tag_pt, base_pt, cv2.RANSAC, 5.0)

            return M