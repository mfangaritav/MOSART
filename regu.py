import numpy as np
import scipy.sparse
import lsquares
import h5py

def makeLineDerivSys(line, starty):
    # Make linear system expressing forward differences for a line
    nr = nc = len(line) + 1
    A = np.zeros((nr, nc), dtype=np.float32)
    b = np.zeros((nr, 1), dtype=np.float32)
    A[0,0] = 1
    b[0] = starty
    for i in range(1,nr):
        A[i,i-1] = -1
        A[i,i] = 1
        b[i] = line[i-1]
        
    return A, b

def makeRegX(grid):
    # Make index grid of output dem (easier book keeping)
    demgr = np.zeros((grid.shape[0], grid.shape[1]+1), dtype=np.int32)
    for i in range(demgr.shape[1]):
        demgr[0, i] = i
        
    for i in range(1, demgr.shape[0]):
        demgr[i,:] = demgr[i-1,:] + demgr.shape[1]
        
    # Make regularization grid
    nr = demgr.shape[0]*(demgr.shape[1]-2)
    nc = demgr.shape[0]*(demgr.shape[1])
    Arx = scipy.sparse.lil_array((nr, nc), dtype=np.float32)
    ri = 0
    for i in range(grid.shape[0]):
        for j in range(1, demgr.shape[1]-1):
            Arx[ri, demgr[i,j-1]] = 1
            Arx[ri, demgr[i, j]] = -2
            Arx[ri, demgr[i, j+1]] = 1
            ri += 1
            
    return Arx
    
def makeRegY(grid):
    # Make index grid of output dem (easier book keeping)
    demgr = np.zeros((grid.shape[0], grid.shape[1]+1), dtype=np.int32)
    for i in range(demgr.shape[1]):
        demgr[0, i] = i
        
    for i in range(1, demgr.shape[0]):
        demgr[i,:] = demgr[i-1,:] + demgr.shape[1]
        
    # Make regularization grid
    nr = (demgr.shape[0]-2)*demgr.shape[1]
    nc = demgr.shape[0]*(demgr.shape[1])
    Ary = scipy.sparse.lil_array((nr, nc), dtype=np.float32)
    ri = 0
    for i in range(1,grid.shape[0]-1):
        for j in range(demgr.shape[1]):
            Ary[ri, demgr[i-1,j]] = 1
            Ary[ri, demgr[i, j]] = -2
            Ary[ri, demgr[i+1, j]] = 1
            ri += 1
            
    return Ary

def makeSystem(grid, lamy = 1, lamx = 1, startys=None):
    As = []
    bs = []
    
    if(startys is None):
        startys = np.zeros(grid.shape[0])
        
    for i in range(grid.shape[0]):
        A, b = makeLineDerivSys(grid[i,:], startys[i])
        As.append(A)
        bs.append(b)
        
    # Assemble As
    Ablks = []
    for i, A in enumerate(As):
        Ablks.append([None]*len(As))
        Ablks[i][i] = A
    Agr = scipy.sparse.bmat(Ablks)

    # Assemble bs
    bgr = np.vstack(bs)
    
    if(lamx == 0 and lamy == 0):
        return Agr, bgr

    astack = [Agr]
    bstack = [bgr]
    
    # X regularization
    if(lamx != 0):
        xreg = makeRegX(grid)*lamx
        astack.append(xreg)
        bstack.append(np.zeros((xreg.shape[0],1)))
        
    # Y regularization
    if(lamy != 0):
        yreg = makeRegY(grid)*lamy
        astack.append(yreg)
        bstack.append(np.zeros((yreg.shape[0],1)))   
        
    # Add on regularization
    Agr = scipy.sparse.vstack(astack)
    bgr = np.vstack(bstack)

    return Agr, bgr

def getdemreg(key, lamx=0.005, lamy=0.5, rangeCrop=None):
    # get gradient
    grad = lsquares.getgrad(key)
    
    if(rangeCrop is not None):
        grad = grad[:,rangeCrop[0]:rangeCrop[1]]
    
    # set nans to zero, but keep mask
    mask = np.isnan(grad)
    grad[np.isnan(grad)] = 0
    
    # Get arcticdem elevation at edge of image as starting point
    # should improve this. maybe can use c?
    
    # Arctic dem
    fd = h5py.File("./descending.h5", mode="r")
    if(rangeCrop is None):
        arc = fd[key]["dem"][:]
    else:
        arc = fd[key]["dem"][:,rangeCrop[0]:rangeCrop[1]]
    fd.close()
    
    # Solve system
    Agr, bgr = makeSystem(grad, lamx=lamx, lamy=lamy, startys=arc[:,0])
    print("System generated for " + key)
    x = scipy.sparse.linalg.lsqr(Agr, bgr)[0].reshape(grad.shape[0], grad.shape[1]+1)
    print("Solved " + key)

    # Crop off extra column (should fix linear system so it doesn't add this)
    x = x[:,1:]
    
    # Apply nan mask
    x[mask] = np.nan
    
    return x