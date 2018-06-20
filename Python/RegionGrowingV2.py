import numpy as np
import cv2 as cv

#Handles everything around RegionGrowing as Validation, multi layer
#in: A filtered Imagestack with the images ready to segmentate
#leakCorrection - Whether the segmentation should go thorugh a second time and catch leaks
#out:the segmented image stack with 1 for a pixel thrown away, 0 for  a nontaken pixel, 255 for a segmentpixel
def RGHandler(imgstack,startindex, startseed, leakCorrection):
    seeds = startseed #our seed Interatio
    index = startindex #our pointer/Cursor to the current image in the stacks
    outputStack = imgstack.copy() #segmented image stack with 1 for considered pixel, 255 for taken pixel
    estimationKernel = 6
    #initialize the result stack
    i = 0
    while (i < len(imgstack)):
        outputStack[i] = np.zeros_like(imgstack[i])
        i+=1

    #Perform the actual RegionGrowing on the first slide
    segSlide = RegionGrowing(imgstack[index],seeds,estimationKernel)
    outputStack[index] = segSlide

    print("Go UP!")
    cont = 1
    while(index < len(segSlide)-1 and cont):
        index += 1
        # Get the points we will use to compare the regions
        # if we have no points to compare the provided area was too small
        Line, Borders = GetComparisonLine(segSlide, 1, 1,estimationKernel)
        if (len(Line) == 0):
            cont = 0
            print("End of Line up: No new points found")
        else:
            # compute new seedpoint with the comparison line on the processed filtered image in comparison to the original image
            seeds = GetNewSeeds(imgstack[index - 1], imgstack[index], Line)
            if (len(seeds) <= 0):
                cont = 0
                print("End of Line Up: Valid Seeds")
            else:
                print("onSlide: ",index)
                segSlide = RegionGrowing(imgstack[index], seeds, estimationKernel)
                if (segSlide[0,0] == -1):
                    cont = 0
                # add Borders for the next iteration
                if (leakCorrection):
                    for sp in Borders:
                        imgstack[index - 1][sp[0], sp[1]] = 0
                # show me the Borders
                # for sp in Borders:
                #     slidestack[index -1][sp[0], sp[1]] = 200

                # show me the comparison line
                # for sp in Line:
                #     currslide[sp[0], sp[1]] = 140
                #     for spp in fourNeighbours(sp,imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 140

                # show me the seed points
                for sp in seeds:
                    segSlide[sp[0], sp[1]] = 80
                    for spp in EightNeighbours(sp, imgstack[index].shape):
                        segSlide[spp[0], spp[1]] = 80
                #TODO: Validation
                outputStack[index] = segSlide

    print("Go DOWN")
    index = startindex+1
    cont = 1
    segSlide = outputStack[index]
    while (index > 1 and cont):
        index -= 1
        # Get the points we will use to compare the regions
        # if we have no points to compare the provided area was too small
        Line, Borders = GetComparisonLine(segSlide, 1, 1,estimationKernel)
        if (len(Line) == 0):
            cont = 0
            print("End of Line down: No new points found")
        else:
            # compute new seedpoint with the comparison line on the processed filtered image in comparison to the original image
            seeds = GetNewSeeds(imgstack[index + 1], imgstack[index], Line)
            if (len(seeds) <= 0):
                cont = 0
                print("End of Line down: No valid new seed")
            else:
                # compute the average of the givven seeds
                print("On Slide", index )
                segSlide = RegionGrowing(imgstack[index], seeds, estimationKernel)
                if (segSlide[0,0] == -1):
                    cont = 0

                # add Borders for the next iteration
                if (leakCorrection):
                    for sp in Borders:
                        imgstack[index + 1][sp[0], sp[1]] = 0

                ##show me the Borders
                # for sp in Borders:
                #     slidestack[index +1][sp[0], sp[1]] = 200
                # show me the comparison line
                # for sp in Line:
                #     currslide[sp[0], sp[1]] = 140
                #     for spp in fourNeighbours(sp, imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 140

                # show me the seed points
                for sp in seeds:
                    segSlide[sp[0], sp[1]] = 80
                    for spp in EightNeighbours(sp, imgstack[index].shape):
                        segSlide[spp[0], spp[1]] = 80

                #morphologisches schließen
                kernelsizeMorph = 10
                kernel = np.ones((kernelsizeMorph, kernelsizeMorph), np.uint8)
               # segSlide = cv.morphologyEx(segSlide, cv.MORPH_CLOSE, kernel)
                outputStack[index] = segSlide
    if (leakCorrection):
        return RGHandler(imgstack, startindex, startseed,0)
    else:
        return outputStack

    return outputStack


#Performs a Region Growing algorithm on the provided image by estimating his treshhold for every seed and checking the
#difference between the average of the surrounding and the possible pixel
#in: a filtered image and the seedpoints to start the spread; also the kernel for the estimation in width
#out: the single segmented image
def RegionGrowing(img,seeds,eKernel):
    segmentImg = np.zeros_like(img)
    # everything with the seeds; 0,1 point coordinates; 2 average of the seed surrounding; 3 upper bound; 4 lower bound
    seedData = []
    toBeChecked = [] #list with elements that could be possbile segmented points; 1,2 point coordinates; 3 originSeed
    processed = 0
    #initialize the seeds
    counter = 0
    for s in seeds:
        avg,sdevo,sdevu = GetAverageAndSDeviation(img,s,eKernel)
        seedData.append((s[0],s[1],avg,sdevo*1.5,sdevu*1.5))
        #print(counter.__str__()+" unten"+sdevu.__str__()+"oben:"+sdevo.__str__())
        toBeChecked.append((s[0],s[1],counter))
        counter +=1
    #perform the regionGrowing
    while(len(toBeChecked) >0):
        p = toBeChecked[0]
        #point was already considered?
        if ((segmentImg[p[0],p[1]] != p[2] and segmentImg[p[0],p[1]] != 255) or segmentImg[p[0],p[1]] == 0):
            origVal = img[p[0],p[1]]
            currAvg = seedData[p[2]][2]
            #upper or lower bound
            if (origVal > currAvg):
                if(origVal-currAvg <= seedData[p[2]][3]):
                    #we found a positive pixel
                    segmentImg[p[0],p[1]] = 255
                    processed+=1
                    for n in EightNeighbours((p[0],p[1]),img.shape):
                        toBeChecked.append((n[0],n[1],p[2]))
                else:
                    segmentImg[p[0],p[1]] = p[2]
            else:
                #lower
                if (currAvg - origVal<= seedData[p[2]][4]):
                    # we found a positive pixel
                    segmentImg[p[0], p[1]] = 255
                    processed+=1
                    for n in EightNeighbours((p[0],p[1]), img):
                        toBeChecked.append((n[0], n[1], p[2]))
                else:
                    segmentImg[p[0], p[1]] = p[2]

        toBeChecked.pop(0)
        if (processed > (176*176/3)):
            cont = 0
            print("ausgelaufen die Kacke")
            segmentImg[0,0] = -1
    return segmentImg

# returns the Average and Standard Deviation of the provided point an his kernel surrounding
#in: point inside the image and a kernel bigger than 3
def GetAverageAndSDeviation(img, point,kernel):
    average = 0
    SDevD = 0
    SDevU = 0
    uppers = 1
    downers = 1
    cursor = int(kernel/2)
    values = []
    #calculate average
    for i in range(point[0]-cursor,point[0]+cursor+1):
        for j in range(point[1]-cursor,point[1]+cursor+1):
            values.append(img[i,j])
            average += img[i,j]
    average /= (kernel*kernel)

    #calculate Standard Deviation
    for v in values:
        if (average > v):
            SDevU += average-v
            downers +=1
        else:
            SDevD += v-average
            uppers+=1
    SDevD /= uppers
    SDevU /= downers

    return average,SDevD,SDevU


#calculates a list of all pixel in a 8N surrounding
#in: point inside the image, validate outer shape of the image
#out: eightNeighbours inside the image
def EightNeighbours(point,shape):
    maximum = shape[0] - 1
    minimum = shape[1] - 1
    List = []
    if (isinstance(point,tuple)):
        #if (point[0] == shape[0]-1 or point[1] == shape[1]-1):
            #List.append((point[0],point[1]))
            #return List
        List.append(((point[0] + 1), point[1]))
        List.append(((point[0] - 1), point[1]))
        List.append((point[0], (point[1] + 1)))
        List.append((point[0], (point[1] - 1)))
        List.append(((point[0] + 1), point[1] + 1))
        List.append(((point[0] + 1), point[1] - 1))
        List.append((point[0] - 1, (point[1] + 1)))
        List.append((point[0] - 1, (point[1] - 1)))

    return List

####################################################
################NEXT SLIDE FUNCTIONS################

#Computes the new Seedpoint
#input: Source Image and TargetImage for comparison along the comparison line
#out: returns the best fitting point as new seed point
def GetNewSeeds(imgsource,imgtarget,comparline):
    compars = []
    newseeds = []
    sum = 0
    #compares the intensity in the old image with the intensity in the new image
    for l in comparline:
        intensS,y,y = GetAverageAndSDeviation(imgsource,l,5)
        intensT,y,y = GetAverageAndSDeviation(imgtarget,l,5)
        compars.append(dif(intensT,intensS))
        sum += dif(intensS,intensT)
    #get all seeds that have a minimum distance better than min
    min = sum/len(compars)
    runner = 0
    for c in compars:
        if (c < min):
           newseeds.append(comparline[runner])
        runner += 1
    return newseeds

#returns a list of points that are in the middle of the segmented image as seen from y and x direction
#in: Valid Segementation of an image
#out: set of points that are relativly in the middle and their kernel area is definitly inside the image segment
def GetComparisonLine(res,y,x, kernel):
    i = 4
    output = []
    borders = []
    maxborderlength = 10 #HERE: Stellt hier die Länge der Grenze ein
    rgKernelSize = kernel
    i+=rgKernelSize
    if (x != 0):
        while (i < res.shape[1]-(9+rgKernelSize)):#9 to cut out possible points that are too close to the border
            i+=2
            up = 0
            cont = 1
            inAreaCounter = 0
            inArea = 0
            #
            while (up < res.shape[0]-9 and cont):
                up+=1

                #change from inside the are to outside the area
                if (res[up,i] == 0 and inArea == 1):
                    if (inAreaCounter > maxborderlength):
                        middle = up - int(inAreaCounter/2)
                        #Make Sure we have also up and down more than kernel times space
                        cursor = int(rgKernelSize *-1)
                        success = 1
                        while (cursor < rgKernelSize):
                            if (res[middle,i+cursor] == 0):
                                success = 0
                            cursor+=1
                        #CheckDiagonal
                        if ((res[int(middle+rgKernelSize),int(i+rgKernelSize)] == 0) or
                                (res[int(middle-rgKernelSize),int(i+rgKernelSize)] == 0) or (res[int(middle+rgKernelSize),int(i-rgKernelSize)] == 0) or
                                (res[int(middle-rgKernelSize),int(i-rgKernelSize)] == 0)) :
                            success = 0
                        if (success):
                            output.append((middle, i,inAreaCounter))
                    if (inAreaCounter <= maxborderlength):
                        j = inAreaCounter
                        while (j >0):
                            borders.append((up-j, i))
                            j -= 1
                    inArea = 0
                #change from outside an are to inside an area
                if (res[up,i] != 0 and inArea == 0):
                    inArea = 1
                    inAreaCounter = 1
                #we are staying inside an area
                if(inArea):
                    inAreaCounter +=1

    if (y != 0):
        i = 4
        while (i < res.shape[0] - (9+rgKernelSize)):  # 9 to cut out possible points that are too close to the border
            i += 2
            up = 0
            cont = 1
            inAreaCounter = 0
            inArea = 0

            #
            while (up < res.shape[1] - 9 and cont):
                up += 1
                # change from inside the are to outside the area
                if (res[i, up] == 0 and inArea == 1):
                    inArea = 0
                    if (inAreaCounter > maxborderlength):
                        middle = up - int(inAreaCounter / 2)
                        #Ensure the border also in the other direction
                        cursor = int(maxborderlength/2 *-1)
                        success = 1
                        while (cursor < maxborderlength/2):
                            if (res[i+cursor,middle] == 0):
                                success = 0
                            cursor+=1
                        #CheckDiagonal
                        if ((res[int(middle+maxborderlength/2),int(i+maxborderlength/2)] == 0) or
                                (res[int(middle-maxborderlength/2),int(i+maxborderlength/2)] == 0) or (res[int(middle+maxborderlength/2),int(i-maxborderlength/2)] == 0) or
                                (res[int(middle-maxborderlength/2),int(i-maxborderlength/2)] == 0)) :
                            success = 0
                        if (success):
                            output.append((i, middle,inAreaCounter))

                    if (inAreaCounter <= maxborderlength):
                        j = inAreaCounter
                        while (j >0):
                            borders.append((i,up-j))
                            j -=1
                # change from outside an are to inside an area
                if (res[i, up] != 0 and inArea == 0):
                    inArea = 1
                    inAreaCounter = 0
                # we are staying inside an area
                if (inArea):
                    inAreaCounter += 1

    return output, borders

def dif(x,y):
    res = x-y
    if res <0:
        res *= -1
    return res