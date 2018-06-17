import numpy as np
import time

#Handles everything around RegionGrowing as Validation and multi layer
#maxdif - maximal difference every pixel is allowed to differ from the original seed
#returns the segmented image stack
def RGHandler(imgstack,startindex, startseed, maxdif, takeborders):
    seed = startseed
    index = startindex
    slidestack = imgstack.copy() # to store the results
    cont = 1 # to say whether we already have an error and can continue
    #initialize the result stack
    i = 0
    while (i < len(imgstack)):
        slidestack[i] = np.zeros_like(imgstack[i])
        i+=1
    # start with the first slide and the provided user input
    tresh = 0# Tresh is initialized to zero because it will adapt itself in the RegionGrowing Algortihm

    originalSeedAverage = GetAverageValueSeeds(seed,imgstack[index])
    #actual compute region
    # perform the region growing on the filtered image and evaluate the time
    starttime = time.localtime(time.time())
    currslide = RegionGrowing(imgstack[index],seed,tresh,maxdif,originalSeedAverage)
    endtime = time.localtime(time.time())
    print("Zeit für RG:",endtime[5]-starttime[5],"s")
    # Validate first slide
    while(currslide[0,0] == -1 and cont == 1):
        #currslide = RegionGrowing(imgstack[index],seed,tresh-2,maxdif,originalSeedAverage)
        #TODO: Failhandling
        cont = 0
    # show intial seed
    for sp in seed:
        currslide[sp[0], sp[1]] = 140
        for spp in fourNeighbours(sp, imgstack[index].shape):
            currslide[spp[0], spp[1]] = 140
    # add to finished slides
    slidestack[index] = currslide

    #Perform the same on the following slides
    highBound = len(imgstack)-1 # HERE: Stellt hier ein wie viele Slides er nach oben gehen soll
    print("It is going up!")
    while(index < highBound and cont):
        index += 1
        # Get the points we will use to compare the regions
        # if we have no points to compare the provided area was too small
        Line, Borders = GetComparisonLine(currslide, 1, 1)
        if (len(Line) == 0):
            cont = 0
            print("End of Line up")
        else:
            # compute new seedpoint with the comparison line on the processed filtered image in comparison to the original image
            seed = GetNewSeed(imgstack[index-1], imgstack[index], Line)
            if(len(seed) == 0):
                cont = 0
            else:
                # compute the average of the givven seeds
                originalSeedAverage = GetAverageValueSeeds(seed, imgstack[index])
                tresh = 0# Tresh is initialized to zero because it will adapt itself in the RegionGrowing Algortihm
                print("to go:",(highBound-index))
                currslide = RegionGrowing(imgstack[index], seed, tresh,maxdif,originalSeedAverage)

                # add Borders for the next iteration
                if (takeborders):
                    for sp in Borders:
                        imgstack[index -1][sp[0], sp[1]] = 0
                # show me the Borders
                # for sp in Borders:
                #     slidestack[index -1][sp[0], sp[1]] = 200

                # show me the comparison line
                # for sp in Line:
                #     currslide[sp[0], sp[1]] = 140
                #     for spp in fourNeighbours(sp,imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 140

                # show me the seed points
                # for sp in seed:
                #     currslide[sp[0], sp[1]] = 80
                #     for spp in fourNeighbours(sp,imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 80

                #validate the slide
                while (currslide[0, 0] == -1 and cont):
                    cont = 0
                    currslide = RegionGrowing(imgstack[index], seed, tresh - 2, maxdif, originalSeedAverage)
                    #TODO: Failhandling
                slidestack[index] = currslide

    print("It is going down")
    index = startindex+1
    cont = 1
    lowBound = 1
    currslide = slidestack[index]
    while (index > lowBound and cont):
        index -= 1
        # Get the points we will use to compare the regions
        # if we have no points to compare the provided area was too small
        Line, Borders = GetComparisonLine(currslide,1,1)
        if (len(Line) == 0):
            cont = 0
            print("End of Line down")
        else:
            # compute new seedpoint with the comparison line on the processed filtered image in comparison to the original image
            seed = GetNewSeed(imgstack[index + 1], imgstack[index], Line)
            if (len(seed) == 0):
                cont = 0
            else:
                # compute the average of the givven seeds
                originalSeedAverage = GetAverageValueSeeds(seed, imgstack[index])
                tresh = 0  # Tresh is initialized to zero because it will adapt itself in the RegionGrowing Algortihm
                print("to go:", (index-lowBound))
                currslide = RegionGrowing(imgstack[index], seed, tresh, maxdif, originalSeedAverage)

                #add Borders for the next iteration
                if (takeborders):
                    for sp in Borders:
                     imgstack[index +1][sp[0], sp[1]] = 0

                ##show me the Borders
                # for sp in Borders:
                #     slidestack[index +1][sp[0], sp[1]] = 200
                #show me the comparison line
                # for sp in Line:
                #     currslide[sp[0], sp[1]] = 140
                #     for spp in fourNeighbours(sp, imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 140

                # show me the seed points
                # for sp in seed:
                #     currslide[sp[0], sp[1]] = 80
                #     for spp in fourNeighbours(sp, imgstack[index].shape):
                #         currslide[spp[0], spp[1]] = 80

                # validate the slide
                while (currslide[0, 0] == -1 and cont):
                    cont = 0
                    currslide = RegionGrowing(imgstack[index], seed, tresh - 2, maxdif, originalSeedAverage)
                slidestack[index] = currslide
    if (takeborders):
        return RGHandler(imgstack,startindex,startseed,maxdif,0)
    else:
        return slidestack

#Computes the new Seedpoint
#input: Source Image and TargetImage for comparison along the comparison line
#out: returns the best fitting point as new seed point
def GetNewSeed(imgsource,imgtarget,comparline):
    compars = []
    newseeds = []
    #compares the intensity in the old image with the intensity in the new image
    for l in comparline:
        intensS,y = AverageMinValueEightNeigh(imgsource,l)
        intensT,y = AverageMinValueEightNeigh(imgtarget,l)
        compars.append(dif(intensT,intensS))
    #get all seeds that have a minimum distance better than min
    min = 3
    runner = 0
    for c in compars:
        if (c < min):
            newseeds.append(comparline[runner])
        runner += 1
    return newseeds

def GetComparisonLine(res,y,x):
    i = 4
    output = []
    borders = []
    maxborderlength = 10 #HERE: Stellt hier die Länge der Grenze ein
    if (x != 0):
        while (i < res.shape[1]-9):#9 to cut out possible points that are too close to the border
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
                        output.append((middle, i))
                    elif (inAreaCounter <= maxborderlength):
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
        while (i < res.shape[0] - 9):  # 9 to cut out possible points that are too close to the border
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
                        output.append((i, middle))
                    else:
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

#pure Region Growing Algo
#maxdif - maximal difference every pixel is allowed to differ from the original seed
#returns filtered mask
def RegionGrowing(img, seed,tresh, maxdif, seedAV):
    checkers = [] #points that are in need to be checked
    output = np.zeros_like(img)
    counter = 0
    treshHolds = []
    if (isinstance(seed,list)):
        for sp in seed:
            checkers.append((sp[0],sp[1],counter))
            treshHolds.append(0)
            counter += 1
    else:
        checkers.append((seed[0],seed[1],0))
    cont = 1
    starttime = time.localtime(time.time())
    processed = 0 # counts our processed points for failsafe
    tresh = 0
    alpha = 0.4
    while (len(checkers)>0 and cont):
        current = checkers[0]
        output[current[0], current[1]] = 255

        #The first 16 Pixel around a seed are seen as in the liver. Compute them first
        if (processed <= len(seed)*16):
            for neigh in eightNeighbours(current,img.shape):
                if (output[neigh[0],neigh[1]]==0):
                    dist = dif(img[neigh[0], neigh[1]], img[current[0], current[1]])
                    if (dist < maxdif):
                        output[neigh[0], neigh[1]] = 255
                        checkers.append((neigh[0],neigh[1],current[2]))
                        treshHolds[current[2]] += dif(img[neigh[0],neigh[1]],img[current[0],current[1]])
                        # if (dif(img[neigh[0],neigh[1]],img[current[0],current[1]])> tresh):
                        #     tresh = dif(img[neigh[0],neigh[1]],img[current[0],current[1]])
                    processed += 1
        elif (processed+1 == len(seed)*16):
            counter = 0
            processed += 1
            for tre in treshHolds:
                treshHolds[counter] = tre/16
                counter +=1
        #continue with the unsafe points
        if (processed > len(seed)*16):
            for neighbour in eightNeighbours(current,img.shape): #perform the search for correspondent pixel in 8N
                if (output[neighbour[0], neighbour[1]] == 0):
                    dist = dif(img[neighbour[0], neighbour[1]],img[current[0], current[1]])
                    distToOrgin = dif(img[neighbour[0], neighbour[1]],seedAV)
                    # check whether the new pixel is around our current pixel and in the valuespace around our seed
                    tresh = treshHolds[current[2]]
                    if (dist<=tresh and distToOrgin<maxdif):
                        output[neighbour[0], neighbour[1]] = 255
                        checkers.append((neighbour[0],neighbour[1],current[2]))
                        treshHolds[current[2]] = tresh + alpha*(tresh - dist) # Update tresh on every occasion
                    else:
                        output[neighbour[0], neighbour[1]] = 1
                    processed += 1
        checkers.pop(0)

        #failsafe case RegionGrowing is taking longer than 10s
        currtime = time.localtime(time.time())
        if (currtime[5]-starttime[5] >= 10):
            #cont = 0
            print("RegionGrowing took to long- adjusting")
            #output[0, 0] = -1

        #failsafe case RegionGrowing was too big: Liver is not bigger than 1/3 of the image
        if  processed > ((img.shape[0]*img.shape[1])/3):
            print("tresh too high")
            cont = 0
            #TODO: Failsafe: rekursiv call?
            output[0,0] = -1


    return output


#calculate an Average value for the different seed points
def GetAverageValueSeeds(seeds, img):
    originalAV = 0
    if (isinstance(seeds,list)):
        for sp in seeds:
            originalAV += img[sp[0],sp[1]]
        originalAV /= len(seeds)
    else:
        originalAV = img[seeds[0],seeds[1]]

    return originalAV

#calculate the average value and minimal pixel difference of the pixels of a 7x7 kernel; closer ones are more weighted
def AverageMinValueEightNeigh(img, point):
    aver = 0
    min = 500
    rpoint = point

    for n in eightNeighbours(point, img.shape):
        for n2 in eightNeighbours(n, img.shape):
            for n3 in eightNeighbours(n2, img.shape):
                aver += img[n3[0], n3[1]]
                dist = dif(img[n3[0], n3[1]], img[point[0], point[1]])
                if (dist <min and n3 != point):
                    min = dist
                    rpoint = n3
    aver /= 512

    return aver, rpoint

##Position and Calculation Helpers

def fourNeighbours(seed,shape):
    maximum = shape[0]-1
    minimum = shape[1]-1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0]+1), seed[1]))
    List.append(((seed[0] - 1), seed[1]))
    List.append((seed[0], (seed[1]+1)))
    List.append((seed[0], (seed[1]-1)))

    return List

def eightNDiagonal(seed,shape):
    maximum = shape[0] - 1
    minimum = shape[1] - 1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0] + 1), seed[1] + 1))
    List.append(((seed[0] + 1), seed[1] - 1))
    List.append((seed[0] - 1, (seed[1] + 1)))
    List.append((seed[0] - 1, (seed[1] - 1)))

    return List

def eightNeighbours(seed,shape):
    maximum = shape[0]-1
    minimum = shape[1]-1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0]+1), seed[1]))
    List.append(((seed[0] - 1), seed[1]))
    List.append((seed[0], (seed[1]+1)))
    List.append((seed[0], (seed[1]-1)))
    List.append(((seed[0]+1), seed[1]+1))
    List.append(((seed[0] + 1), seed[1]-1))
    List.append((seed[0]-1, (seed[1]+1)))
    List.append((seed[0]-1, (seed[1]-1)))

    return List

def dif(x,y):
    res = x-y
    if res <0:
        res *= -1
    return res