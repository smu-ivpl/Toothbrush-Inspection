##
# FINAL!
#
###
import cv2, os
import numpy as np
import time
from scipy.signal import find_peaks
from multiprocessing import Process

norm_list = []
pre_err_list = []
err_list_6 = []
err_list_65 = []
post_err_list = []
sum_wpix = []
w_trim = 0
inf_time = []
total_start = time.time()


class SideToothbrush(Process):
    def __init__(self):
        Process.__init__(self)

    def run(self, imgname, _img):
        print(" ############################# t3.py start! #############################")
        
        a_list = []
        b_list = []
        c_list = []

        try:
            if os.path.exists(imgname):
                print("sleep for 2 sec!")
                time.sleep(2)
                image = cv2.imread(imgname)
            else:
                return 0
            #image = imageio.imread(imgname)
            
            image = cv2.resize(image, (700, 500))
        
            img = image.copy()  # contour 좌표를 구하기 위한 원본 복사 이미지
            img1 = image.copy()  # ROI영역을 만들기 위한 원본 복사 이미지1
            img_morph = image.copy()
            img_draw = image.copy()  

            y, x = img.shape[:2]

            # right image trim
            img = img[:, :x - 40]
            img1 = img1[:, :x - 40]
            img_morph = img_morph[:, :x - 40]
            img_draw = img_draw[:, :x - 40]
            h, w = img.shape[:2]
            h1, w1 = img1.shape[:2]

            # w_trim = np.sum(img[:10, :10] == 255)

            # if w_trim >= 3:
            img = img[40:, :]
            img1 = img1[40:, :]
            img_morph = img_morph[40:, :]
            img_draw = img_draw[40:, :]

            h, w = img.shape[:2]
            h1, w1 = img1.shape[:2]

            for y in range(h):
                for x in range(w):
                    if img[y, x][0] < 230 or img[y, x][1] < 230 or img[y, x][2] < 230:
                        # y, x 순서인 이유 : 영상 행렬은 높이, 길이로 저장되므로
                        img[y, x] = 0

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            # preprocessing
            closed = self.preprocessing(gray_img)

            # get minmax
            x, y, w, h, x_min, x_max, y_min, y_max = self.getMinMax(closed)

            # apply binary -> not roi image but whole image!!!
            morph_img = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
            ret, morph_thresh = cv2.threshold(morph_img, 80, 255, cv2.THRESH_BINARY)

            ## morphology
            open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            # 열림 연산
            open = cv2.morphologyEx(morph_thresh, cv2.MORPH_OPEN, open_k)
            opening = open.copy()

            cv2.rectangle(opening, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0, 0, 0), -1)

            iimg_draw = img_draw.copy()
            recc = cv2.rectangle(iimg_draw, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0, 0, 255), 1)
            cv2.imwrite(f"./datasets/side_brush/preprocess_result/preprocess_{_img}", recc)

            # result area
            roi_img = opening[:y + h - 10, :]
            thresh_img = morph_thresh[:y + h - 10, :]

            # 결과 출력
            merged = np.hstack((thresh_img, roi_img))

            # count pixels
            num_wpix = np.sum(roi_img == 255)

            # 이미지에 글자 합성하기
            result = cv2.putText(merged, f"{num_wpix} pixels!", (560, 230), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1,
                                cv2.LINE_AA)

            # cv2.imshow("input image", image)  # 입력이미지 출력
            # cv2.imshow('preprocessing', result)

            # cv2.imwrite(f"./temp/pix_{_img}", result)

            if num_wpix > 18 and num_wpix <= 4000:
                print(f'{_img} is error toothbrush - preprocessing')
                print("############################## t3.py finished ###################################")    
                pre_err_list.append(_img)
                err = 1
                
                return err

            if num_wpix <= 18 or num_wpix > 4000:
                # print(f'{_img} is checking - postprocessing')

                name = _img.split(".")[0]
                ## get pok!

                morph_img_post = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
                ret, morph_thresh_post = cv2.threshold(morph_img_post, 70, 255, cv2.THRESH_BINARY)

                try :
                    # 1. get hole to hole distance.
                    fst_pix, last_pix, f_m, s_m, l_m, distance, trim_handle, size_hole = self.get_hole_distance(morph_thresh_post,
                                                                                                        img_draw, closed)
                    # img_brush_height_fm = morph_thresh[y_min: y_min + 200, int(f_m):w]

                    # 식모 첫 중간 부터 끝 중간 까지 trim 한것
                    brush_mid_to_mid = trim_handle[:, int(f_m): int(l_m)]
                    brush_mid_to_mid_draw = img_draw[:, int(f_m): int(l_m)]

                    # 식모 첫 부터 끝까지 trim 한것
                    brush_f_to_l = trim_handle[:, int(fst_pix): int(last_pix)]
                    brush_f_to_l_draw = img_draw[:, int(fst_pix): int(last_pix)]

                    # brush f to l size
                    b_height, b_width = brush_f_to_l.shape
                    peak_distance = int(b_width / 12)

                    x, y, w, h, x_min, x_max, y_min, y_max = self.getMinMax(brush_mid_to_mid)

                    # 식모 위 끝에 가깝게 trim
                    brush = brush_f_to_l[y_min:y_min + h, :]
                    brush_draw1 = brush_f_to_l_draw[y_min:y_min + h, :]
                    brush_draw2 = brush_draw1.copy()
                    brush_draw3 = brush_draw1.copy()

                    ### count black pixel
                    ob_h, ob_w = brush.shape
                    num_bpix = []

                    for i in range(ob_w):
                        num_bpix.append(np.sum(brush[:, i] == 0))

                    # get peaks
                    peaks, height = find_peaks(num_bpix, distance=peak_distance, height=25)

                    peaks = list(peaks)
                    for p in peaks:
                        if p < int(b_width / 20):
                            peaks.remove(p)

                    # print("new peaks : ", peaks)
                    # cut 할 지점
                    cut = [0]
                    cut = cut + list(peaks)
                    cut.append(last_pix)
                    # print("cutpoint  :", cut)

                    #####
                    # there are two algorithms
                    # a. neighbor diff / hole_distance > 0.2
                    # b. max-min / hole_distance > 0.4
                    #####

                    # hole distance
                    hole_distance = distance

                    min_pix = []
                    min_pix_x = []
                    w_area = []
                    neighbor_diff = []

                    # to get rid of noise on uppoer part
                    erode_brush = brush.copy()
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # 침식 연산 적용 ---②
                    erode_brush = cv2.erode(erode_brush, k)

                    x, y, w, h, x_min, x_max, y_min, y_max = self.getMinMax(erode_brush)

                    hh = int(h / w)

                    if y_min - hh <= 1:
                        bhh = 0
                    else:
                        bhh = y_min - hh

                    wi_hole = []
                    start_x_list = []
                    stack_x = 0
                    start_x_list.append(stack_x)
                    for i in range(10):
                        each_brush = brush[bhh: y_max, cut[i]: cut[i + 1]]
                        # each_brush_draw = brush_draw[:, cut[i]: cut[i+1]]
                        check_break = True
                        e_h, e_w = each_brush.shape

                        check_break = True
                        for a in range(e_h):
                            for b in range(e_w):
                                # print(a,",",b,"=",each_img[a][b])
                                if each_brush[a][b] == 255:
                                    min_pix.append(a)  # append height 좌표값
                                    min_pix_x.append(b + stack_x)  # append width 좌표값
                                    check_break = False
                                    # print(a, ",", b, "=", each_img[a][b])
                                    break
                                if a == e_h and b == e_w:
                                    min_pix.append(30)

                            if check_break == False:
                                break
                        # print("minpix : ", min_pix)
                        if i != 0:
                            neighbor_diff.append(abs(min_pix[i] - min_pix[i - 1]))

                        stack_x += e_w
                        start_x_list.append(stack_x)

                    pad = 30
                    # looking for coordinate of the maximum neighbor_difference
                    max_diff = max(neighbor_diff)
                    index = neighbor_diff.index(max_diff)
                    l_x = start_x_list[index]
                    lr_x = start_x_list[index + 1]
                    r_x = start_x_list[index + 2]

                    left_p_y = min_pix[index]
                    right_p_y = min_pix[index + 1]

                    # looking for coordinate of the maximum max-min difference
                    # The lowest y value
                    max_p_y = max(min_pix)
                    max_index = min_pix.index(max_p_y)
                    s_max_p_x = start_x_list[max_index]
                    e_max_p_x = start_x_list[max_index + 1]

                    # The highest y value
                    min_p_y = min(min_pix)
                    min_index = min_pix.index(min_p_y)
                    s_min_p_x = start_x_list[min_index]
                    e_min_p_x = start_x_list[min_index + 1]

                
                    # a. neighbor diff / hole_distance >= 0.26
                    if round((max(neighbor_diff) / hole_distance), 2) >= 0.26:
                        post_err_list.append(_img)
                        a_list.append(_img)
                        cv2.putText(brush_draw1,
                                    f"{round((max(neighbor_diff) / hole_distance), 2)} >= 0.26 so, ERROR!",
                                    (pad - 10, brush_draw1.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.imshow("postprocessing A type: ", brush_draw1)
                        cv2.imwrite(f"./datasets/side_brush/side_result/error_A/err_A_{_img}", brush_draw1)
                        err = 1
                        return err
                    else:
                        cv2.putText(brush_draw1,
                                    f"{round((max(neighbor_diff) / hole_distance), 2)} < 0.26 so, NORMAL!",
                                    (pad - 10, brush_draw1.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.imshow("postprocessing A type: ", brush_draw1)
                    # b. max-min / hole_distance >= 0.3
                    minmaxdiff = max(min_pix) - min(min_pix)
                    if _img not in post_err_list and (minmaxdiff / hole_distance) >= 0.3:
                        post_err_list.append(_img)
                        b_list.append(_img)
                        cv2.putText(brush_draw2,
                                    f"{round((minmaxdiff / hole_distance), 2)} >= 0.3 so, ERROR!",
                                    (pad - 10, brush_draw2.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.imshow("postprocessing B type: ", brush_draw2)
                        cv2.imwrite(f"./datasets/side_brush/side_result/error_B/err_B_{_img}", brush_draw2)
                        err = 1
                        return err
                    else:
                        cv2.putText(brush_draw2,
                                    f"{round((minmaxdiff / hole_distance), 2)} < 0.3 so, NORMAL!",
                                    (pad - 10, brush_draw2.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.imshow("postprocessing B type: ", brush_draw2)

                        # while (True):
                        #    if cv2.waitKey(1) & 0xFF == ord('x'):
                        #        cv2.destroyAllWindows()
                        #        break

                    # a & b
                    if _img not in post_err_list and (max(neighbor_diff) / hole_distance) >= 0.2 and (
                            minmaxdiff / hole_distance) >= 0.23:
                        post_err_list.append(_img)
                        c_list.append(_img)
                        cv2.putText(brush_draw3,
                                    f"{round((max(neighbor_diff) / hole_distance), 2)} >= 0.2 & {round((minmaxdiff / hole_distance), 2)} >= 0.23 so, ERROR!",
                                    (pad - 10, brush_draw3.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        # cv2.imshow("postprocessing A&B type: ", brush_draw3)
                        cv2.imwrite(f"./datasets/side_brush/side_result/error_AB/err_AB_{_img}", brush_draw3)
                        err = 1
                        return err

                    if _img in post_err_list:
                        print(f'{_img} is error toothbrush - postprocessing')



                except :
                    print('ERROR: this image is hard to detect in side process!')
                    print(" ############################# t3.py finished #############################")
                    return 0


            if _img not in pre_err_list and _img not in post_err_list:
                norm_list.append(_img)
                print(f'{_img} is normal toothbrush - postprocessing')
                print(" ############################# t3.py finished #############################")
                err = 0
                return err
        except:
            
            return 0


    def preprocessing(self, input):
        edged = cv2.Canny(input, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        return closed


    def getMinMax(self, image_c):
        ## contour
        contours, _ = cv2.findContours(image_c.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_xy = np.array(contours)
        # print(contours_xy.shape)

        contours_image = cv2.drawContours(image_c.copy(), contours, -1, (0, 0, 255), 3)
        # cv2.imshow("rr", contours_image)
        # cv2.waitKey(0)
        # cv2.imwrite(f'/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/contour{_img}', contours_image)
        # cv2.imwrite('/home/user/NOAH/GONGIN/datasets/test/CAM2/result/contour.png', contours_image)

        x_min, x_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
                x_min = min(value)
                x_max = max(value)
        # print(x_min)
        # print(x_max)

        # y의 min과 max 찾기
        y_min, y_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
                y_min = min(value)
                y_max = max(value)
        # print(y_min)
        # print(y_max)

        # image trim 하기
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min

        return x, y, w, h, x_min, x_max, y_min, y_max


    def get_hole_distance(self, getimg, img_draw, closed):
        dimg = getimg.copy()
        x, y, w, h, x_min, x_max, y_min, y_max = self.getMinMax(closed)
        trim_handle = dimg[:y_max - 10, :]

        # trim_h_draw = img_draw[:y_max - 10, :]

        pre_trim_h, pre_trim_w = trim_handle.shape

        # for eliminate to bar
        trim_pix = 0
        for p in range(pre_trim_h):
            if trim_handle[p][0] == 255:
                # print("f", f)
                trim_pix = p
                break

        trim_handle = dimg[:trim_pix - 10, :]

        # trim_h_draw = img_draw[:trim_pix - 10, :]

        h_height, h_width = trim_handle.shape

        # cv2.imwrite('/home/user/NOAH/GONGIN/datasets/test/CAM2/result/trim.png', trim_handle)
        # cv2.imshow("tt", trim_handle)
        # cv2.waitKey(0)

        for f in range(h_width):
            if trim_handle[h_height - 1][f] == 255:
                # print("f", f)
                fst_pix = f
                break

        for i in range(fst_pix, h_width - 1):
            if trim_handle[h_height - 1][i] == 0:
                fst_e_pix = i
                # print("fst_e_pix", fst_e_pix)
                break

        for s in range(fst_e_pix, h_width - 1):
            if trim_handle[h_height - 1][s] == 255:
                sec_pix = s
                break

        ## for last brush!
        for l in reversed(range(h_width)):
            if trim_handle[h_height - 1][l] == 255:
                # print("last_e", l)
                last_pix = l
                break
        last_trim = trim_handle[:, :l]
        for l in reversed(range(last_trim.shape[1])):
            if trim_handle[h_height - 1][l] == 0:
                # print("last_ee", l)
                last_se = l
                break

        size_hole = (fst_e_pix - fst_pix)
        f_m = fst_pix + (size_hole / 2)
        s_m = sec_pix + (size_hole / 2)
        l_m = last_se + ((last_pix - last_se) / 2)
        distance = (s_m - f_m)
        # print(f_m)
        # print(s_m)

        # 선그리기
        # cv2.line(trim_h_draw, (int(f_m), h_height - 50), (int(f_m), h_height - 50), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
        # cv2.line(trim_h_draw, (int(s_m), h_height - 50), (int(s_m), h_height - 50), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow('image line', trim_h_draw)
        # cv2.waitKey(0)

        # cv2.imwrite(f"/home/ivpl-d28/Pycharmprojects/NOAH/dataset/hole_distance/hole_distance_{_img}", trim_h_draw)

        return fst_pix, last_pix, f_m, s_m, l_m, distance, trim_handle, size_hole

def side_brush(**kwargs):

    in_que1= kwargs['que_in_1']
    out_que1= kwargs['que_out_1']
    in_que2 = kwargs['que_in_2'] 
    out_que2 = kwargs['que_out_2'] 
    in_que = kwargs['que_in_3'] 
    out_que = kwargs['que_out_3'] 
    in_que4 = kwargs['que_in_4'] 
    out_que4 = kwargs['que_out_4']
    CAM1 = kwargs['cam1']
    CAM2 = kwargs['cam2']
    CAM3 = kwargs['cam3']
    tmp_q = in_que
    
    while not kwargs['stop_event'].wait(1e-9):
        if in_que.qsize() > 0:
            
        
            image_path= in_que.pop()

            if not os.path.exists(image_path):
                continue
            else:
                _img = image_path.split("/")[-1]
                
                result = kwargs['smodel'].run(image_path, _img)
                
                cam1_num = int(_img.split("_")[0]) - 5
                cam3_num = int(_img.split("_")[0]) + 5
                cam1_dir = os.path.join(CAM1+str(cam1_num).zfill(5)+"_&Cam1Img.bmp")
                cam3_dir = os.path.join(CAM3+str(cam3_num).zfill(5)+"_&Cam3Img.bmp")
    
                
                if result:
                    if os.path.exists(image_path):
                        os.rename(image_path, image_path.split('.')[0] + '_0010.png')
                        if os.path.exists(cam1_dir):
                            os.rename(cam1_dir, cam1_dir.split('.')[0] + '_0010.png')
                        if os.path.exists(cam3_dir):
                            os.rename(cam3_dir, cam3_dir.split('.')[0] + '_0010.png')
                    
                else:
                    out_que.put(cam1_dir)

