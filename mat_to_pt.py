import torch
import scipy.io

def mat_to_pt(filename, variablename): #filename and variablename must be strings e.g. 'filename.mat' , 'variablename'
    mat = scipy.io.loadmat(filename)
    pt = torch.tensor(mat[variablename],dtype=torch.cfloat)
    return pt