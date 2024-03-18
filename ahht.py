import argparse
import math
import numpy as np
import skimage.io
import skimage.transform

N = 256

def GetValleys(H) :
    n = len(H)
    valleys = []
    if H[0][0] < H[1][0] :
        valleys.append(H[0][1])
    if H[0][0] == H[1][0] :
        k = 1
        while H[0][0] == H[k][0]:
            k += 1
        k -= 1
        if H[k + 1][0] > H[k][0]:
            valleys.append(H[k // 2][1])
    flag = True
    i = 1
    while i < n-1 :
        if (H[i][0] < H[i - 1][0]) and (H[i][0] < H[i + 1][0]) :
            valleys.append(H[i][1])
        elif H[i][0] < H[i - 1][0] :
            k = 1
            while k + i < n and H[i][0] == H[i + k][0] :
                k += 1
            k -= 1
            if k + i == n - 1 :
                valleys.append(H[i + k // 2][1])
                flag = False
            elif H[i + k][0] < H[i + 1 + k][0] :
                valleys.append(H[i + k // 2][1])
            i += k
        i += 1
    if flag == True and H[n - 1][0] < H[n - 2][0] :
        valleys.append(H[n - 1][1])
    return valleys
def GetNextHist(H, valleys, w) :
    if len(valleys) == 0 :
        return H
    new_H = []
    left = 0
    right = 0
    index = 0
    if valleys[0] == H[0][1]:
        index = 1
    for r in range (index, len(valleys)) :
        for m in range (len(H)) :
            (h, l, lR) = H[m]
            if l == valleys[r] :
                right = m
                break
        if H[right][1] - H[left][1] <= w :
            bin = GetMergeBin(H, left, right)
            new_H.append(bin)
        else :
            bins = GetMergeSetofBins(H, left, right, w)
            new_H.extend(bins)
        if new_H[len(new_H) - 1] == H[right] : 
            left = right
            new_H.pop()
        else : left = right + 1
    if left < len(H):
        if H[len(H) - 1][1] - H[left][1] <= w:
            bin = GetMergeBin(H, left, len(H) - 1)
            new_H.append(bin)
        else :
            bins = GetMergeSetofBins(H, left, len(H) - 1, w)
            new_H.extend(bins)
    return new_H
def GetMergeBin(H, left, right) :
    count = 0
    int = 0
    num = 0
    for i in range (left, right + 1) : 
        count += H[i][0]
        num += 1
        int += H[i][1]
    bin = (count, int // num, H[right][2])
    return bin
def GetMergeSetofBins(H, left, right, w) :
    bins = []
    for i in range (left, right + 1) : 
        bins.append(H[i])
    min_index = FindMinDistance(bins)
    a = bins[min_index + 1][1] - bins[min_index][1]
    while (a <= w) :
        bins_1 = []
        for i in range(min_index) :
            bins_1.append(bins[i])
        bins_1.append((bins[min_index][0] + bins[min_index + 1][0],
                       (bins[min_index][1] + bins[min_index + 1][1]) // 2,
                       bins[min_index + 1][2]))
        for i in range (min_index + 2, len(bins)) :
            bins_1.append(bins[i])
        bins = bins_1
        min_index = FindMinDistance(bins)
        if min_index == -1 : break
        a = bins[min_index + 1][1] - bins[min_index][1]
    return bins
def FindMinDistance(bins) :
    min_index = -1
    if (len(bins) == 1): return min_index
    min_distance = 256
    for i in range(len(bins) - 1) :
        if bins[i + 1][1] - bins[i][1] < min_distance :
            min_distance = bins[i + 1][1] - bins[i][1]
            min_index = i
    return min_index
def Dist(arr) :
    min_dist = N
    c1 = -1
    c2 = -1
    length = len(arr)
    for i in range (0, length) :
        for j in range (i + 1, length):
            if max(abs(int(arr[j][0][0]) - int(arr[i][0][0])),
                   abs(int(arr[j][0][1]) - int(arr[i][0][1])),
                   abs(int(arr[j][0][2]) - int(arr[i][0][2]))) < min_dist:
                min_dist = max(abs(int(arr[j][0][0]) - int(arr[i][0][0])),
                               abs(int(arr[j][0][1]) - int(arr[i][0][1])),
                               abs(int(arr[j][0][2]) - int(arr[i][0][2])))
                c1 = i
                c2 = j
    return (min_dist, c1, c2)
def CalcSpan(n, H) :
    left = 0
    right = 0
    sum1 = 0
    sum2 = 0
    for i in range(255, -1, -1) :
        sum1 += H[i][0]
        if sum1 / n > 0.01:
            left = i
            break
    for i in range(0, N) :
        sum2 += H[i][0]
        if sum2 / n > 0.01 :
            right = i
            break
    return left - right
def GetAHH(H1, w) :
    Hists = []
    Hists.append(H1)
    valleys = GetValleys(H1)
    H2 = GetNextHist(H1, valleys, w)
    while H1 != H2 :
        Hists.append(H2)
        H1 = H2
        valleys = GetValleys(H1)
        H2 = GetNextHist(H1, valleys, w)
    return Hists
def DistGrey(arr) :
    min_dist = N
    c1 = -1
    c2 = -1
    length = len(arr)
    for i in range (0, length) :
        for j in range (i + 1, length):
            if abs(int(arr[j]) - int(arr[i])) < min_dist:
                min_dist = abs(int(arr[j]) - int(arr[i]))
                c1 = i
                c2 = j
    return (min_dist, c1, c2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')  
    parser.add_argument('parameters', nargs='*') #w, Td
    parser.add_argument('input_file')
    parser.add_argument('output_file_initial')
    parser.add_argument('output_file_merged',)
    args = parser.parse_args()

img = skimage.io.imread(args.input_file)
height, width = img.shape[:2]
w = int(args.parameters[0])
general_number = height * width
Nd = general_number // 1000
Td = int(args.parameters[1])
if args.command == 'color':
    #Creating basic arrays corresponding to the histograms of the 1st level
    H1 = [0] * N
    H2 = [0] * N
    H3 = [0] * N
    for i in range (height) :
        for j in range (width) :
            H1[img[i][j][0]] += 1
            H2[img[i][j][1]] += 1
            H3[img[i][j][2]] += 1
    #histograms of the 1st level
    H_first_1, H_first_2, H_first_3 = [], [], []
    for i in range(N) :
        H_first_1.append((H1[i], i, i))
        H_first_2.append((H2[i], i, i))
        H_first_3.append((H3[i], i, i))
    span1 = CalcSpan(general_number, H_first_1)
    span2 = CalcSpan(general_number, H_first_2)
    span3 = CalcSpan(general_number, H_first_3)
    #Creating hierarchical histograms
    Hists1 = GetAHH(H_first_1, w * span1 / N)
    Hists2 = GetAHH(H_first_2, w * span2 / N)
    Hists3 = GetAHH(H_first_3, w * span3 / N)
    #histograms of the last level
    H1 = Hists1[len(Hists1) - 1]
    H2 = Hists2[len(Hists2) - 1]
    H3 = Hists3[len(Hists3) - 1]
    #initial segmentation
    for i in range (height) :
        for j in range (width) :
            color1 = img[i][j][0]
            color2 = img[i][j][1]
            color3 = img[i][j][2]
            for k in range(len(H1)):
                if color1 <= H1[k][2] :
                    img[i][j][0] = H1[k][1]
                    break
            for k in range(len(H2)):
                if color2 <= H2[k][2] :
                    img[i][j][1] = H2[k][1]
                    break
            for k in range(len(H3)):
                if color3 <= H3[k][2] :
                    img[i][j][2] = H3[k][1]
                    break
    res = np.clip(img, 0, 255)
    res = res.astype(np.uint8)
    skimage.io.imsave(args.output_file_initial, res)
    #find all the colors corresponding to homogeneous areas in the picture.
    arr = []
    for i in range (height) :
        for j in range (width) :
            (color1, color2, color3) = img[i][j]
            flag = True
            for k in range (len(arr)) :
                if arr[k][0] == (color1, color2, color3):
                    count = arr[k][1]
                    arr[k] = ((color1, color2, color3), count + 1)
                    flag = False
                    break
            if flag == True:
                arr.append(((color1, color2, color3), 1))
    #looking for all areas with a number of pixels less than Nd
    small_areas = []
    for i in range(len(arr)) :
        if arr[i][1] < Nd :
            small_areas.append(arr[i])
    #merging small areas with the nearest ones
    while len(small_areas) != 0 :
        ((color1, color2, color3), number) = small_areas[0]
        small_areas.remove(((color1, color2, color3), number))
        arr_index = -1
        min_distance = N
        for i in range (len(arr)) :
            if arr[i] != ((color1, color2, color3), number) and max(abs(int(arr[i][0][0]) - int(color1)),
                                                                    abs(int(arr[i][0][1]) - int(color2)),
                                                                    abs(int(arr[i][0][2]) - int(color3))) < min_distance :
                min_distance = max(abs(int(arr[i][0][0]) - int(color1)),
                                   abs(int(arr[i][0][1]) - int(color2)),
                                   abs(int(arr[i][0][2]) - int(color3)))
                arr_index = i
        if any([item == arr[arr_index] for item in small_areas]) :
            small_areas.remove(arr[arr_index])
        count = arr[arr_index][1]
        count += number
        arr[arr_index] = (arr[arr_index][0], count)
        base_elem = arr[arr_index]
        arr.remove(((color1, color2, color3), number))
    
        if base_elem[1] < Nd :
            small_areas.append(base_elem)
        for i in range (height) :
            for k in range (width) :
                if img[i][k][0] == color1 and img[i][k][1] == color2 and img[i][k][2] == color3 :
                    img[i][k][0] = base_elem[0][0]
                    img[i][k][1] = base_elem[0][1]
                    img[i][k][2] = base_elem[0][2]
    #merge the nearest areas with each other
    while (a := Dist(arr))[0] < Td :
        i1 = a[1]
        i2 = a[2]
        (color1, num1) = arr[i1]
        (color2, num2) = arr[i2]
        arr.remove((color1, num1))
        arr.remove((color2, num2))
        arr.append(((num1 * color1[0] // (num1 + num2) + num2 * color2[0] // (num1 + num2),
                     num1 * color1[1] // (num1 + num2) + num2 * color2[1] // (num1 + num2),
                     num1 * color1[2] // (num1 + num2) + num2 * color2[2] // (num1 + num2)), num1 + num2))
        for i in range (height) :
            for k in range (width) :
                if img[i][k][0] == color1[0] and img[i][k][1] == color1[1] and img[i][k][2] == color1[2] :
                    img[i][k][0] = arr[len(arr) - 1][0][0]
                    img[i][k][1] = arr[len(arr) - 1][0][1]
                    img[i][k][2] = arr[len(arr) - 1][0][2]
                if img[i][k][0] == color2[0] and img[i][k][1] == color2[1] and img[i][k][2] == color2[2] :
                    img[i][k][0] = arr[len(arr) - 1][0][0]
                    img[i][k][1] = arr[len(arr) - 1][0][1]
                    img[i][k][2] = arr[len(arr) - 1][0][2]
    res = np.clip(img, 0, 255)
    res = res.astype(np.uint8)
    skimage.io.imsave(args.output_file_merged, res)
if args.command == 'grey' :
    if len(img.shape) == 3:
        img = img[:, :, 0]
    H = [0] * N
    for i in range (N) :
        H[i] = 0
    for i in range (height) :
        for j in range (width) :
            H[img[i][j]] += 1
    H_first = []
    for i in range(N) :
        H_first.append((H[i], i, i))
    Hists = GetAHH(H_first, w)
    H = Hists[len(Hists) - 1]
    for i in range (height) :
        for j in range (width) :
            color = img[i][j]
            for k in range(len(H)):
                if color <= H[k][2] :
                    img[i][j] = H[k][1]
                    break
    res = np.clip(img, 0, 255)
    res = res.astype(np.uint8)
    skimage.io.imsave(args.output_file_initial, res)
    arr = []
    for i in range (height) :
        for j in range (width) :
            color = img[i][j]
            flag = True
            for k in range (len(arr)) :
                if arr[k][0] == color:
                    count = arr[k][1]
                    arr[k] = (color, count+1)
                    flag = False
                    break
            if flag == True:
                arr.append((color, 1))
    small_areas = []
    for i in range(len(arr)) :
        if arr[i][1] < Nd :
            small_areas.append(arr[i])
    while len(small_areas) != 0 :
        (color, number) = small_areas[0]
        small_areas.remove((color, number))
        arr_index = -1
        min_distance = N
        for i in range (len(arr)) :
            if arr[i] != (color, number) and abs(int(arr[i][0]) - int(color)) < min_distance :
                min_distance = abs(int(arr[i][0]) - int(color))
                arr_index = i
        if any([item == arr[arr_index] for item in small_areas]) :
            small_areas.remove(arr[arr_index])
        count = arr[arr_index][1]
        count += number
        arr[arr_index] = (arr[arr_index][0], count)
        base_elem = arr[arr_index]
        arr.remove((color, number))
    
        if base_elem[1] < Nd :
            small_areas.append(base_elem)
        else :
            for i in range (height) :
                for k in range (width) :
                    if img[i][k] == color :
                        img[i][k] = base_elem[0]
    arr = [tup[0] for tup in arr]
    while (a := DistGrey(arr))[0] < Td :
        i1 = a[1]
        i2 = a[2]
        color1 = arr[i1]
        color2 = arr[i2]
        arr.remove(color1)
        arr.remove(color2)
        arr.append(min(color1, color2) + abs(int(color1)-int(color2)) // 2)
        for i in range (height) :
            for k in range (width) :
                if img[i][k] == color1 or img[i][k] == color2 :
                    img[i][k] = arr[len(arr)-1]
    res = np.clip(img, 0, 255)
    res = res.astype(np.uint8)
    skimage.io.imsave(args.output_file_merged, res)
