##
# FINAL!
#
###
import cv2, os
import numpy as np
import time
import matplotlib.pyplot as plt
from findpeaks import findpeaks
from scipy.signal import find_peaks
import csv

image_path = './datasets/side_brush'
dirs = os.listdir(image_path)
# print(dirs)
images = [file for file in dirs if file.endswith('.png') or file.endswith('.bmp')]
images.sort()
print("how many images :", len(images))

norm_list = []
pre_err_list = []
err_list_6 = []
err_list_65 = []
post_err_list = []
sum_wpix = []
w_trim = 0
inf_time = []
total_start = time.time()

def preprocessing(input):
    edged = cv2.Canny(input, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return closed


def getMinMax(image_c):
    ## contour
    contours, _ = cv2.findContours(image_c.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_xy = np.array(contours)
    # print(contours_xy.shape)

    contours_image = cv2.drawContours(image_c.copy(), contours, -1, (0, 0, 255), 3)
    #cv2.imshow("rr", contours_image)
    #cv2.waitKey(0)
    #cv2.imwrite(f'/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/contour{_img}', contours_image)

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


def get_hole_distance(getimg, img_draw):

    dimg = getimg.copy()
    x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(closed)
    trim_handle = dimg[:y_max - 10, :]

    trim_h_draw = img_draw[:y_max - 10, :]
    h_height, h_width = trim_handle.shape

    #cv2.imshow("tt", trim_handle)
    #cv2.waitKey(0)
    for f in range(h_width):
        if trim_handle[h_height - 1][f] == 255:
            #print("f", f)
            fst_pix = f
            break

    for i in range(fst_pix, h_width - 1):
        if trim_handle[h_height - 1][i] == 0:
            fst_e_pix = i
            #print("fst_e_pix", fst_e_pix)
            break

    for s in range(fst_e_pix, h_width - 1):
        if trim_handle[h_height - 1][s] == 255:
            #print("sec", s)
            sec_pix = s
            break

    ## for last brush!
    for l in reversed(range(h_width)):
        if trim_handle[h_height - 1][l] == 255:
            #print("last_e", l)
            last_pix = l
            break
    last_trim = trim_handle[:, :l]
    for l in reversed(range(last_trim.shape[1])):
        if trim_handle[h_height - 1][l] == 0:
            #print("last_ee", l)
            last_se = l
            break

    size_hole = (fst_e_pix - fst_pix)
    f_m = fst_pix + (size_hole / 2)
    s_m = sec_pix + (size_hole / 2)
    l_m = last_se + ((last_pix - last_se) / 2)
    distance = (s_m - f_m)
    #print(f_m)
    #print(s_m)

    # 선그리기
    #cv2.line(trim_h_draw, (int(f_m), h_height - 50), (int(f_m), h_height - 50), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
    #cv2.line(trim_h_draw, (int(s_m), h_height - 50), (int(s_m), h_height - 50), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
    #cv2.imshow('image line', trim_h_draw)
    #cv2.waitKey(0)

    #cv2.imwrite(f"/home/ivpl-d28/Pycharmprojects/NOAH/dataset/hole_distance/hole_distance_{_img}", trim_h_draw)

    return fst_pix, last_pix, f_m, s_m, l_m, distance, trim_handle, size_hole


a_list = []
b_list = []
c_list = []

#f = open('noah_standard.csv', 'w', newline='')
#final = open('final.csv', 'w', newline='')

for _img in images:
    imgname = os.path.join(image_path, _img)

    total_start = time.time()
    image = cv2.imread(imgname)
    image = cv2.resize(image, (700, 500))
    # cv2.imshow("input image", image)  # 입력이미지 출력
    # cv2.waitKey(30)
    
    img = image.copy()  # contour 좌표를 구하기 위한 원본 복사 이미지
    img1 = image.copy()  # ROI영역을 만들기 위한 원본 복사 이미지1
    img_morph = image.copy()
    img_draw = image.copy() # copy for check and draw
    # cv2.imshow('result_image', image)
    # cv2.waitKey(0)

    y, x = img.shape[:2]

    # right image trim
    img = img[:, :x - 40]
    img1 = img1[:, :x - 40]
    img_morph = img_morph[:, :x - 40]
    img_draw = img_draw[:, :x - 40]
    h, w = img.shape[:2]
    h1, w1 = img1.shape[:2]

    #w_trim = np.sum(img[:10, :10] == 255)

    #if w_trim >= 3:
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

    # cv2.imshow('result_image', gray_img)
    # cv2.waitKey(0)

    # preprocessing
    closed = preprocessing(gray_img)
    # cv2.imshow('result_image', gray_img)
    # cv2.waitKey(0)

    # get minmax
    x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(closed)

    # apply binary -> not roi image but whole image!!!
    morph_img = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
    ret, morph_thresh = cv2.threshold(morph_img, 80, 255, cv2.THRESH_BINARY)

    ## morphology
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # 열림 연산
    open = cv2.morphologyEx(morph_thresh, cv2.MORPH_OPEN, open_k)
    opening = open.copy()

    cv2.rectangle(opening, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0, 0, 0), -1)


    iimg_draw = img_draw.copy()
    recc = cv2.rectangle(iimg_draw, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0, 0, 255), 1)
    cv2.imwrite(f"./datasets/side_brush/preprocess_result/preprocess_{_img}", recc)
    # cv2.imshow("preprocessing : ", recc)
    # cv2.waitKey(3)


    # result area
    roi_img = opening[:y + h - 10, :]
    thresh_img = morph_thresh[:y + h - 10, :]

    # 결과 출력
    merged = np.hstack((thresh_img, roi_img))

    # count pixels
    num_wpix = np.sum(roi_img == 255)

    # 이미지에 글자 합성하기
    result = cv2.putText(merged, f"{num_wpix} pixels!", (560, 230), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

    # cv2.imshow("input image", image)  # 입력이미지 출력
    # cv2.imshow('preprocessing', result)

    #cv2.imwrite(f"./temp/pix_{_img}", result)

    start = time.time()
    if num_wpix > 18 and num_wpix <= 4000:
        print(f'{_img} is error toothbrush - preprocessing')
        pre_err_list.append(_img)


    if num_wpix <= 18 or num_wpix > 4000:
        #print(f'{_img} is checking - postprocessing')

        name = _img.split(".")[0]
        ## get pok!

        morph_img_post = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
        ret, morph_thresh_post = cv2.threshold(morph_img_post, 70, 255, cv2.THRESH_BINARY)

        # 1. get hole to hole distance.
        fst_pix, last_pix, f_m, s_m, l_m, distance, trim_handle, size_hole = get_hole_distance(morph_thresh_post, img_draw)
        #img_brush_height_fm = morph_thresh[y_min: y_min + 200, int(f_m):w]

        # 식모 첫 중간 부터 끝 중간 까지 trim 한것
        brush_mid_to_mid = trim_handle[: , int(f_m): int(l_m)]
        brush_mid_to_mid_draw = img_draw[: , int(f_m): int(l_m)]

        # 식모 첫 부터 끝까지 trim 한것
        brush_f_to_l = trim_handle[:, int(fst_pix): int(last_pix)]
        brush_f_to_l_draw = img_draw[:, int(fst_pix): int(last_pix)]

        # brush f to l size
        b_height,b_width = brush_f_to_l.shape
        peak_distance = int(b_width / 12)
        ## to check
        #cv2.imshow("tt", brush_f_to_l_draw)
        #cv2.waitKey(0)

        x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(brush_mid_to_mid)

        # 식모 위 끝에 가깝게 trim
        brush = brush_f_to_l[y_min:y_min+h, :]
        brush_draw1 = brush_f_to_l_draw[y_min:y_min+h, :]
        brush_draw2 = brush_draw1.copy()
        brush_draw3 = brush_draw1.copy()
        #cv2.imshow("tt", brush_draw)
        #cv2.waitKey(0)


        ### count black pixel
        ob_h, ob_w = brush.shape
        num_bpix = []

        for i in range(ob_w):
            num_bpix.append(np.sum(brush[:,i] == 0))

        # get peaks
        peaks, height = find_peaks(num_bpix, distance = peak_distance, height=25)

        # draw graph
        #plt.plot(num_bpix)
        #plt.plot(peaks, height['peak_heights'], 'x')
        #plt.savefig(f'/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_graphs/{name}_peak_graph.png')

        #plt.show()
        #print("peaks : ", peaks)
        peaks = list(peaks)
        for p in peaks:
            if p < int(b_width / 20):
                peaks.remove(p)

        #print("new peaks : ", peaks)
        #cut 할 지점
        cut = [0]
        cut = cut + list(peaks)
        cut.append(last_pix)
        #print("cutpoint  :", cut)

       #####
        # there are two algorithms
        # a. neighbor diff / hole_distance > 0.2
        # b. max-min / hole_distance > 0.4
        #####

        # hole distance
        hole_distance = distance

        min_pix = []
        min_pix_x = []
        w_area =[]
        neighbor_diff = []

        # to get rid of noise on uppoer part
        erode_brush = brush.copy()
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 침식 연산 적용 ---②
        erode_brush = cv2.erode(erode_brush, k)

        x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(erode_brush)

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
            each_brush = brush[bhh: y_max, cut[i]: cut[i+1]]
            #each_brush_draw = brush_draw[:, cut[i]: cut[i+1]]
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
            #print("minpix : ", min_pix)
            if i != 0:
                neighbor_diff.append(abs(min_pix[i] - min_pix[i - 1]))
                
            stack_x += e_w
            start_x_list.append(stack_x)



        # for i in range(9):
        #    cv2.line(brush_draw, ((w_list[i] + cut[i]), (min_pix[i]+bhh)), ((w_list[i+1] + cut[i+1]),(min_pix[i+1]+bhh)) , (0,0,255), 2)
        

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
        
        brush_draw1 = cv2.copyMakeBorder(
            brush_draw1,
            top=pad,
            bottom=80,
            left=pad,
            right=pad,
            borderType=cv2.BORDER_CONSTANT
        )
        
        brush_draw2 = cv2.copyMakeBorder(
            brush_draw2,
            top=pad,
            bottom=80,
            left=pad,
            right=pad,
            borderType=cv2.BORDER_CONSTANT
        )

        brush_draw3 = cv2.copyMakeBorder(
            brush_draw3,
            top=pad,
            bottom=80,
            left=pad,
            right=pad,
            borderType=cv2.BORDER_CONSTANT
        )

        cv2.line(brush_draw1, (l_x + pad, left_p_y + pad), (lr_x + pad, left_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw1, (lr_x + pad, right_p_y + pad), (r_x + pad, right_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw1, (int(f_m) + pad - fst_pix, brush_draw1.shape[0] - 80), (int(f_m) + pad - fst_pix, brush_draw1.shape[0] - 80), (0, 0, 255), 5)
        cv2.line(brush_draw1, (int(s_m) + pad - fst_pix, brush_draw1.shape[0] - 80), (int(s_m) + pad - fst_pix, brush_draw1.shape[0] - 80), (0, 0, 255), 5)
    
        cv2.putText(brush_draw1, f"gap: {max(neighbor_diff)}",  # abs(left_p_y - right_p_y)}",
                             (0, pad), cv2.FONT_HERSHEY_PLAIN, 2,
                             (255, 0, 255), 1,
                             cv2.LINE_AA)
        cv2.putText(brush_draw1, f"A rate: {max(neighbor_diff)}/{hole_distance} = {round((max(neighbor_diff) / hole_distance), 2)}",
                             (pad, brush_draw1.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2,
                             (255, 255, 255), 1,
                             cv2.LINE_AA)
                             
        # cv2.imwrite(f"/home/yjkim/NOAH/gongin/datasets/side_brush/post_result/graph_A_{_img}", brush_draw1)

        cv2.line(brush_draw2, (s_max_p_x + pad, max_p_y + pad), (e_max_p_x + pad, max_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw2, (s_min_p_x + pad, min_p_y + pad), (e_min_p_x + pad, min_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw2, (int(f_m) + pad - fst_pix, brush_draw2.shape[0] - 80), (int(f_m) + pad - fst_pix, brush_draw2.shape[0] - 80), (0, 0, 255), 5)
        cv2.line(brush_draw2, (int(s_m) + pad - fst_pix, brush_draw2.shape[0] - 80), (int(s_m) + pad - fst_pix, brush_draw2.shape[0] - 80), (0, 0, 255), 5)
        
        cv2.putText(brush_draw2, f"gap: {max(min_pix) - min(min_pix)}",  # {max_p_y - min_p_y}", 
                             (0, pad), cv2.FONT_HERSHEY_PLAIN, 2, 
                             (255, 0, 255), 1,
                             cv2.LINE_AA)
        cv2.putText(brush_draw2, f"B rate: {max(min_pix) - min(min_pix)}/{hole_distance} = {round(((max(min_pix) - min(min_pix)) / hole_distance), 2)}",
                             (pad, brush_draw2.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2, 
                             (255, 255, 255), 1,
                             cv2.LINE_AA)
        
        cv2.line(brush_draw3, (l_x + pad, left_p_y + pad), (lr_x + pad, left_p_y + pad), (0, 127, 255))
        cv2.line(brush_draw3, (lr_x + pad, right_p_y + pad), (r_x + pad, right_p_y + pad), (0, 127, 255))
        cv2.line(brush_draw3, (s_max_p_x + pad, max_p_y + pad), (e_max_p_x + pad, max_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw3, (s_min_p_x + pad, min_p_y + pad), (e_min_p_x + pad, min_p_y + pad), (255, 0, 255))
        cv2.line(brush_draw3, (int(f_m) + pad - fst_pix, brush_draw2.shape[0] - 80),
                 (int(f_m) + pad - fst_pix, brush_draw3.shape[0] - 80), (0, 0, 255), 5)  
        cv2.line(brush_draw3, (int(s_m) + pad - fst_pix, brush_draw2.shape[0] - 80),
                 (int(s_m) + pad - fst_pix, brush_draw3.shape[0] - 80), (0, 0, 255), 5)

        cv2.putText(brush_draw3, f"A gap: {max(neighbor_diff)}",  # abs(left_p_y - right_p_y)}",
                    (0, pad), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(brush_draw3,
                    f"A rate: {max(neighbor_diff)}/{hole_distance} = {round((max(neighbor_diff) / hole_distance), 2)}",
                    (pad, brush_draw3.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.putText(brush_draw3, f", B gap: {max(min_pix) - min(min_pix)}",  # {max_p_y - min_p_y}", 
                    (int(brush_draw3.shape[1]/2 - 10), pad), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 127, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(brush_draw3,
                    f"B rate: {max(min_pix) - min(min_pix)}/{hole_distance} = {round(((max(min_pix) - min(min_pix)) / hole_distance), 2)}",
                    (pad, brush_draw3.shape[0] - 35), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

        # cv2.imwrite(f"/home/yjkim/NOAH/gongin/datasets/side_brush/post_result/graph_B_{_img}", brush_draw2)
        
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
        else:
            cv2.putText(brush_draw2,
                        f"{round((minmaxdiff / hole_distance), 2)} < 0.3 so, NORMAL!",
                        (pad - 10, brush_draw2.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 1,
                        cv2.LINE_AA)
            # cv2.imshow("postprocessing B type: ", brush_draw2)

            #while (True):
            #    if cv2.waitKey(1) & 0xFF == ord('x'):
            #        cv2.destroyAllWindows()
            #        break
        
        # a & b
        if _img not in post_err_list and (max(neighbor_diff) / hole_distance) >= 0.2 and (minmaxdiff / hole_distance) >= 0.23:
            post_err_list.append(_img)
            b_list.append(_img)
            cv2.putText(brush_draw3,
                        f"{round((max(neighbor_diff) / hole_distance), 2)} >= 0.2 & {round((minmaxdiff / hole_distance), 2)} >= 0.23 so, ERROR!",
                        (pad - 10, brush_draw3.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1,
                        cv2.LINE_AA)
            # cv2.imshow("postprocessing A&B type: ", brush_draw3)
            cv2.imwrite(f"./datasets/side_brush/side_result/error_AB/err_AB_{_img}", brush_draw3)

        if _img in post_err_list:
          print(f'{_img} is error toothbrush - postprocessing')


        #print("10 Min pixels : ", min_pix)
        #print("neighbor diff : ", neighbor_diff)

        ### check
        #print('홀간높이차(max): ', max(neighbor_diff))
        #print('홀간거리:', hole_distance)
        #print('홀간높이: ', max(neighbor_diff) / hole_distance)

        #print('최대차이: ', minmaxdiff)
        #print('최대차이비율: ', minmaxdiff / hole_distance)

        #wr = csv.writer(f)
        #wr.writerow([_img ,str(max(neighbor_diff)), str(hole_distance), str(max(neighbor_diff) / hole_distance), str(minmaxdiff),  str(minmaxdiff / hole_distance)])

        # cv2.imshow("postprocessing A type: ", brush_draw1)
        # cv2.imshow("postprocessing B type: ", brush_draw2)

    '''
    cv2.imshow("input image", image)  # 입력이미지 출력
    cv2.imshow('preprocessing', result)

    
    while (True):
        if cv2.waitKey(1) & 0xFF == ord('x'):
            cv2.destroyAllWindows()
            break
    '''
    end = time.time()
    inf_time.append(end - start)
    

    if _img not in pre_err_list and _img not in post_err_list:
        norm_list.append(_img)
        print(f'{_img} is normal toothbrush - postprocessing')
    '''
    wr_final = csv.writer(final)
    if _img in pre_err_list:
        wr_final.writerow([_img , "pre_error"])
    elif _img in post_err_list:
        wr_final.writerow(([_img, "post_err"]))
    else:
        wr_final.writerow(([_img, "normal"]))
    '''
    


## get accuracy
TP = 0
TN = 0
n_cnt = 0
e_cnt = 0
error = pre_err_list + post_err_list
submission = {}


for e in error:
    submission[e] = 0

for n in norm_list:
    submission[n] = 1

for i in submission:
    if "normal" in i:
        n_cnt +=1 
    if "normal" not in i:
        e_cnt +=1 
    if "normal" in i and  submission[i] == 1:
        TP += 1
    if "normal" not in i and  submission[i] == 0:
        TN += 1

acc = (TN + TP) / len(images)

print("submission", len(submission))
print("len inference_time (322): ", len(inf_time))
avg_time = sum(inf_time) / len(images)
print("avg_time length : ", len(inf_time))
print("preprocess err : ", len(pre_err_list))
print("postprocess err : ", len(post_err_list))  # <= 7
print("total err : ", len(error))
print("normal list : ", len(norm_list))
print("================================================================")
print("True Positive : ", TP)
print("True Negative : ", TN)
# print(f"(TP + TN) / 전체 이미지 수 : ({TP} + {TN}) / {len(images)}")
print("Accuracy : ", round(acc, 2) * 100 , "%")
print("Average Time per Image : ", round(avg_time, 2), "s")
#f.close()
#final.close
