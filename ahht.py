import argparse
import itertools
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
        if H[right][1] - H[left][1] < w :
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
        if H[len(H) - 1][1] - H[left][1] < w:
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
    a = bins[min_index+1][1] - bins[min_index][1]
    while (a < w) :
        bins_1 = []
        for i in range(min_index) :
            bins_1.append(bins[i])
        bins_1.append((bins[min_index][0] + bins[min_index + 1][0],
                       (bins[min_index][1] + bins[min_index + 1][1]) // 2,
                       bins[min_index+1][2]))
        for i in range (min_index + 2, len(bins)) :
            bins_1.append(bins[i])
        bins = bins_1
        min_index = FindMinDistance(bins)
        if min_index == -1 : break
        a = bins[min_index+1][1] - bins[min_index][1]
    return bins
def FindMinDistance(bins) :
    min_index = -1
    if (len(bins) == 1): return min_index
    min_distance = 256
    for i in range(len(bins) - 1) :
        if bins[i+1][1] - bins[i][1] < min_distance :
            min_distance = bins[i+1][1] - bins[i][1]
            min_index = i
    return min_index
def Dist(arr) :
    min_dist = N
    c1 = -1
    c2 = -1
    length = len(arr)
    
    for (i, point1), (j, point2) in itertools.combinations(enumerate(arr), 2):
        dist = max(abs(int(point2[0][k]) - int(point1[0][k])) for k in range(3))
        if dist != 0 and dist < min_dist:
            min_dist = dist
            c1 = i
            c2 = j
    '''
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
    '''
    return (min_dist, c1, c2)
def CalcSpan(n, H) :
    left = 0
    right = 0
    sum1 = 0
    sum2 = 0
    for i in range(255, -1, -1) :
        #if H[i][0] != 0 :
        #    right = i
        sum1 += H[i][0]
        if sum1 / n > 0.01:
            left = i
            break
    for i in range(0, N) :
        #if H[i][0] != 0 :
        #    left = i
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
def MergeSmallAreas(img, Nd, arr) :
    small_areas = []
    for i in range(len(arr)) :
        if arr[i][1] < Nd :
            small_areas.append(arr[i])
    while len(small_areas) != 0 :
        ((color1, color2, color3), number) = small_areas[0]
        small_areas.remove(((color1, color2, color3), number))
        arr_index = -1
        min_distance = N
        color = (color1, color2, color3)
        '''
        for i in range (len(arr)) :
            if arr[i][0] != (color1, color2, color3) and max(abs(int(arr[i][0][0]) - int(color1)),
                                                             abs(int(arr[i][0][1]) - int(color2)),
                                                             abs(int(arr[i][0][2]) - int(color3))) < min_distance :
                min_distance = max(abs(int(arr[i][0][0]) - int(color1)),
                                   abs(int(arr[i][0][1]) - int(color2)),
                                   abs(int(arr[i][0][2]) - int(color3)))
                arr_index = i
        '''
        for i, point in itertools.filterfalse(lambda x: x[1][0] == (color1, color2, color3), enumerate(arr)):
            dist = max(abs(int(point[0][k]) - int(color[k])) for k in range(3))
            if dist < min_distance:
                min_distance = dist
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
        indices = np.where(np.all(img == (color1, color2, color3), axis=-1))
        img[indices] = base_elem[0]  
    return img
def MergeNearestAreas(img, Td, arr) :
    while (a := Dist(arr))[0] < Td :
        i1 = a[1]
        i2 = a[2]
        ((color11, color12, color13), num1) = arr[i1]
        ((color21, color22, color23), num2) = arr[i2]
        arr.remove(((color11, color12, color13), num1))
        arr.remove(((color21, color22, color23), num2))
        arr.append(((num1 * color11 // (num1 + num2) + num2 * color21 // (num1 + num2),
                     num1 * color12 // (num1 + num2) + num2 * color22 // (num1 + num2),
                     num1 * color13 // (num1 + num2) + num2 * color23 // (num1 + num2)), num1 + num2))
        indices = np.where(np.all(img == (color11, color12, color13), axis=-1))
        img[indices] = arr[len(arr) - 1][0]
        indices = np.where(np.all(img == (color21, color22, color23), axis=-1))
        img[indices] = arr[len(arr) - 1][0]
    return img
def GetSegmentedImage(img, w, Td) :
    height, width = img.shape[:2]
    pixels_number = height * width
    Nd = pixels_number // 1000
    H1 = [0] * N
    H2 = [0] * N
    H3 = [0] * N
    for i in range (height) :
        for j in range (width) :
            H1[img[i, j, 0]] += 1
            H2[img[i, j, 1]] += 1
            H3[img[i, j, 2]] += 1
    H_first_1, H_first_2, H_first_3 = [], [], []
    for i in range(N) :
        H_first_1.append((H1[i], i, i))
        H_first_2.append((H2[i], i, i))
        H_first_3.append((H3[i], i, i))
    span1 = CalcSpan(pixels_number, H_first_1)
    span2 = CalcSpan(pixels_number, H_first_2)
    span3 = CalcSpan(pixels_number, H_first_3)
    
    Hists1 = GetAHH(H_first_1, w*span1/N)
    Hists2 = GetAHH(H_first_2, w*span2/N)
    Hists3 = GetAHH(H_first_3, w*span3/N)

    H1 = Hists1[len(Hists1) - 1]
    H2 = Hists2[len(Hists2) - 1]
    H3 = Hists3[len(Hists3) - 1]
    #print(len(H1))
    #print(len(H2))
    #print(len(H3))
    
    for k in range(len(H1)):
        if k == 0:
            mask = np.logical_and.reduce((img[:,:,0] >= 0, img[:,:,0] <= H1[k][2]))
        else:
            mask = np.logical_and.reduce((img[:,:,0] > H1[k-1][2], img[:,:,0] <= H1[k][2]))
        img[mask, 0] = H1[k][1]
    for k in range(len(H2)):
        if k == 0:
            mask = np.logical_and.reduce((img[:,:,1] >= 0, img[:,:,1] <= H2[k][2]))
        else:
            mask = np.logical_and.reduce((img[:,:,1] > H2[k-1][2], img[:,:,1] <= H2[k][2]))
        img[mask, 1] = H2[k][1]
    for k in range(len(H3)):
        if k == 0:
            mask = np.logical_and.reduce((img[:,:,2] >= 0, img[:,:,2] <= H3[k][2]))
        else:
            mask = np.logical_and.reduce((img[:,:,2] > H3[k-1][2], img[:,:,2] <= H3[k][2]))
        img[mask, 2] = H3[k][1]
    
    res1 = np.clip(img, 0, 255)
    res1 = res1.astype(np.uint8)
    
    color_values, counts = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    a1 = np.empty((len(color_values)), dtype=object)
    a1[:] = [tuple(i) for i in color_values]
    colors_sequence = list(zip(a1, counts))
    
    print(len(colors_sequence))
    
    img = MergeSmallAreas(img, Nd, colors_sequence)
    
    img = MergeNearestAreas(img, Td, colors_sequence)
    
    print(len(colors_sequence))
    res2 = np.clip(img, 0, 255)
    res2 = res2.astype(np.uint8)
    return (res1, res2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', nargs='*') #w, Td
    parser.add_argument('input_file')
    parser.add_argument('output_file_initial')
    parser.add_argument('output_file_merged',)
    args = parser.parse_args()


w = int(args.parameters[0])
Td = int(args.parameters[1])
img = skimage.io.imread(args.input_file)
init, merg = GetSegmentedImage(img, w, Td)
skimage.io.imsave(args.output_file_initial, init)
skimage.io.imsave(args.output_file_merged, merg)
'''
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
'''
