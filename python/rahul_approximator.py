import numpy as np

## Calculates everything in step 2 and returns the scaling value
## param A = an array of PSD Matrices 
## param epsilon = epsilon value
def calScalingValue(A, epsilon): 
    n = len(A)  
    hNorm = 0  
    
    ## Finds what ||h|| is and uses the formula Alp made
    for i in range(n - 1):  
        distance = np.linalg.norm(A[i] - A[i+1])  
        hNorm += distance ** 2 
    distance = np.linalg.norm(A[n-1] - A[0])  
    hNorm += distance ** 2 

    ## Find the scaled value 
    c = hNorm * epsilon ** 2  
    scalingValue = 1 / c
    
    return scalingValue


## Finds h(x) and returns ||h(x)||
## param x = a vector
## param n = size
## param A = n array of PSD matrices
def gethNorm(x, n, A):
    h =[]
    for i in range(n - 1):
        partA = ((np.linalg.norm(np.dot(A[i], x))) ** 2) - (np.linalg.norm(np.dot(A[i+1], x )) ** 2)
        h.append(partA)
    lastPart = ((np.linalg.norm(np.dot(A[n - 1], x))) ** 2) - (np.linalg.norm(np.dot(A[0], x )) ** 2)
    h.append(lastPart)
    hNorm = np.linalg.norm(h)
    
    return hNorm
    

## This is to generate the PSD matrices 
## param n = size
def generatePSDmatrices(n):
    PSDmatrices = []
    for _ in range(n):
        B = np.random.default_rng().normal(size=(n, n)) ##Gaussian 
        A = np.dot(B.T, B)
        eigenvalues, _ = np.linalg.eig(A)
        PSDmatrices.append(A)

    return PSDmatrices


## This is to generate the vectors
## param n = size
def generateVectors(n):
    guassianVectors = []
    for _ in range(n):
        vector = np.random.default_rng().normal(size=(n,1))
        guassianVectors.append(vector)
        
    return guassianVectors


## This is to generate one vector from gaussian distrubutation
## param rowSize = row size of the vector
## param colSize = col size of the vector
def getGaussianVector(rowSize, colSize):
    vector = np.random.default_rng().normal(size=(rowSize, colSize))

    return vector


## Get Gamma Value
## param n = size
def getGamma(n):
    rangeValue = int(input('Please enter your preferred range for gamma, must be a number from 1 - 18 '))
    gamma = rangeValue * np.log(n)
    
    return gamma


## Calculate the Expression
## gamma = gamma value
## param n = size
## param epsilon = epsilon value
## param setOfY = set of vectors
## param PSDmatrices = n number of PSD matrices
def calculateExpression(gamma, n, epsilon, setOfY, PSDmatrices):
    computeSum = 0      
    setLength = len(setOfY)
    for i in range (setLength):
        hNormY = gethNorm(setOfY[i], n, PSDmatrices)
        computeSum += np.exp(gamma * n * np.log(2 / (epsilon ** 2)) * (0.5 - (hNormY ** 2)))
    
    x = computeSum / setLength
    return x


## returns the set of vectors
## param n = size
## param PSDmatrices = n number of PSD matrices
def getSetOfY(n, PSDmatrices):
    
    ## Step 0 - Find a vector v such that ||h(v)|| > n
    v = getGaussianVector(n,1)
    hNormOfV = gethNorm(v, n, PSDmatrices)
    while hNormOfV <= n:
        v = getGaussianVector(n,1)
        hNormOfV = gethNorm(v, n, PSDmatrices)
    
    ## Copy v to all items in queue
    queue = []
    qSize = 5 ## you can change this
    for _ in range (qSize):
        queue.append(v)
    
    ## Initalize the Set S (as per logic in email)
    setOfY = []
    
    ## Step 4.3 - Iterate k times for calculating the setOfY values
    s = 0
    k = np.exp(qSize) + 1 ## k > e^qSize
    iterationCount = 0
    while s < k:
        iterationCount += 1
        y = getGaussianVector(n,1) ## Get gaussian vector
        
        ## repeat until we find a y that satisfies <y,x> < n/2 for all vectors x in queue
        i = 0
        while i < (len(queue)):
            x = queue[i]
            i += 1
            if np.vdot(y,x) >= n/2: ## Using vdot() to get scaler value 
                y = getGaussianVector(n,1)
                i = 0
        hNormY = gethNorm(y, n, PSDmatrices)
    
        ## if ||h(y)|| < n then add it to setOfY
        if hNormY < n: 
            setOfY.append(y)
            s += 1
        else:
            ## remove first item from the que and push y into queue
            np.roll(queue, 1, axis = 0)
            queue[qSize-1] = y 

    print ("iteration Count for calculating setOfY= ", iterationCount)
    return setOfY


## Calculate the approximate value
## gamma = gamma value
## param n = size
## param epsilon = epsilon value
## param PSDmatrices = n number of PSD matrices
def approximate(gamma, n, epsilon, PSDmatrices):
    s = 3    ## This can be changed as needed
    x = 0
    for _ in range (s):
        setOfY = getSetOfY(n, PSDmatrices)
        x += calculateExpression(gamma, n, epsilon, setOfY, PSDmatrices)
  
    return x/s

## Printing matrices
## param matrices = an array of matrices to be printed
def printMatrices(matrices):
    for i, A in enumerate(matrices):
        print(f"A {i+1}:")
        print(A)
        print()


## Asking users for their input and generating
n = int(input('Please enter the number of matrices: '))
epsilon = n ** -2
PSDmatrices = generatePSDmatrices(n)
gamma = getGamma(n)
approxVal = approximate(gamma, n, epsilon, PSDmatrices)
print("approx is = ", approxVal)