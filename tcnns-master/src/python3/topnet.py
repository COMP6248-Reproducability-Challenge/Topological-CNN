import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from scipy.integrate import quad
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift
import sklearn.preprocessing
import copy
import matplotlib.pyplot as plt
import cv2

def get_topnet_func(func_name):
    func_name=[x for x in map(str.strip, func_name.split('.')) if x]
    func=globals()[func_name.pop(0)]
    while func_name:
        func = getattr(func, func_name.pop(0))
    return(func)

def generate_network(device, imageDim, k, convLayerSpecs,
                     activations, pools, weightGrids, freezeLayers, convGrids, lr=1e-3):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convnets = nn.ModuleList()
            for spec in convLayerSpecs:
                self.convnets.append(nn.Conv2d(spec[0], spec[1], spec[2]))

            x= torch.randn(imageDim).view(-1,1,imageDim[0],imageDim[1])
            self._to_linear = None
            self.convs(x)

            self.fc1 = nn.Linear(self._to_linear, 512)
            self.fc2 = nn.Linear(512, k)

        def convs(self, x):
            for i,convnet in enumerate(self.convnets):
                if(activations[i]):
                    x = get_topnet_func(activations[i])(convnet(x))
                else:
                    x = convnet(x)
                if(pools[i]):
                    x = get_topnet_func(pools[i][0])(x,pools[i][1])

            if self._to_linear is None:
                self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

            return x

        def forward(self, x):
            x = self.convs(x)
            x = x.view(-1, self._to_linear)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.softmax(x, dim=1)

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Set initial weights
    if weightGrids:
        for g, grids in enumerate(weightGrids):
            if np.any(grids):
                with torch.no_grad():
                    if copy==0: adjustment = 0
                    for i,x in enumerate(grids):
                        net.convnets[g].weight[i] = torch.nn.Parameter(torch.tensor(x)).to(device)

    # Set convolutional memberships
    if convGrids:
        for g, grids in enumerate(convGrids):
            if np.any(grids):
                tensor_grids = torch.tensor(grids).to(device)
                with torch.no_grad():
                    for i,x in enumerate(tensor_grids):
                        net.convnets[g].weight[i] = net.convnets[g].weight[i]*x


    # Freeze layers
    if np.any(freezeLayers):
        for layer, value in enumerate(freezeLayers):
            net.convnets[layer].requires_grad = not(value)


    return(net,optimizer)


def small_gabor_set(k):
    sigs=list(range(1,4))
    thes=list(range(1,4))
    lams=[10]
    gams=[0,10]
    phis=list(range(0,4))
    grids = np.array([cv2.getGaborKernel((k,k),sig, the, lam, gam, phi, cv2.CV_32F) for sig in sigs for the in thes for lam in lams for gam in gams for phi in phis])
    return(grids)


def generate_circle_membership_grids(rows, cols, nnd):
    def define_l2_distance(grid, center):
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                coord = np.absolute(np.subtract((i,j),center))
                grid[i,j] = max(coord)
        return grid

    def define_distance(grid, center):
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                coord = np.subtract((i,j),center)            
                grid[i,j] = np.sqrt(np.sum(np.square(coord)))
        return grid

    center = math.floor(rows/2),math.floor(cols/2) # Floor because of 0-Index
    distance_grid = define_l2_distance(np.zeros([rows,cols]), center)
    perimeter_grid = distance_grid==np.max(distance_grid)

    convolutional_membership_grids = []
    for i, row in enumerate(perimeter_grid):
        for j, item in enumerate(row):
            if(item):
                grid_copy= ((define_distance(np.zeros([rows,cols]), (i,j))<=nnd)*1)*perimeter_grid
                convolutional_membership_grids.append(grid_copy)

    return(convolutional_membership_grids)

def formula(x,y,th1,th2):
    assert(th1 < np.pi and th1 >= 0)
    assert(th2 < 2*np.pi and th2 >= 0)

    def q2(a):
        return(a)
#         return ( np.sqrt(12) * (a - 0.5) ) #np.sqrt(1.5) * a
    def q3(a):
        return(((2*a - 1)**2) - 1)
#         return((3*(a**2)-1)/2)
#         return (np.sqrt(1.25) * (3.0 * a**2 - 1 ))  # np.sqrt(5.0/8.0) * (3.0 * a**2 - 1 )

    return np.sin(th2)*q2(np.cos(th1)*x + np.sin(th1)*y) + np.cos(th2)*q3(np.cos(th1)*x + np.sin(th1)*y)

def legendre_integrand(x, y1, y2, th1, th2):
    def int_3(y, x, th1, th2):
        return formula(x,y,th1,th2)

    I = quad(int_3, y1, y2, args=(x, th1, th2))
    return I[0]

def legendre_klein_bottle(num_th1=8, num_th2=8, width=5, thresh=None):
    area = (1.0/width) ** 2
    angles1 = [ float(i * np.pi) / num_th1 for i in range(num_th1) ]
    angles2 = [ float(i * 2*np.pi) / num_th1 for i in range(num_th2) ]
    list_weights = []
    for ti in range(len(angles1)):
        for tj in range(len(angles2)):
            th1, th2 = angles1[ti], angles2[tj]
            M = np.zeros([width, width])
            for i in range(width):
                for j in range(width):
                    x1 = i * 1.0 / width
                    x2 = x1 + 1.0 / width
                    y1 = j * 1.0 / width
                    y2 = y1 + 1.0 / width
                    I = quad(legendre_integrand, x1, x2, args=(y1, y2, th1, th2))
                    value = I[0] #/ area
                    M[j,i] = value
            M = M - np.mean(M)
            M = M / np.std(M.flatten())
            if thresh:
                M[M>=thresh]=1
                M[M<thresh]=0
            list_weights.append(M)
    return np.array(list_weights)

def primary_circle_integrand(x, k_y, r, m):
    return max( min( r*(x-(m/2.0)) + m/2.0, k_y + 1), k_y ) - k_y

def rotate(matrix, degree):
    degree = int(degree)
    if abs(degree) not in [0, 90, 180, 270, 360]:
        print("ERROR")
        assert(False)
    if degree == 0:
        return matrix
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree-90)
    else:
        return rotate(zip(*matrix)[::-1], degree+90)

def primary_circle(m, prim_circle_num, thresh=None):
    tot_angle=2*math.pi
    STD = .1
    list_weights = []
    for w in range(0, int(prim_circle_num)):
        M = np.zeros([int(m), int(m)])
        w = float(w)
        theta = w * (tot_angle) / prim_circle_num
        r = math.sin(theta) / math.cos(theta)

        for x_i in range(0, int(m)):
            for y_i in range(0, int(m)):
                I = quad(primary_circle_integrand, x_i, x_i+1, args=(y_i, r, m))
                sign = np.sign(r)
                if sign == 0: sign = 1
                A = sign * I[0]
                M[y_i, x_i] = A

        if theta > math.pi:
            M = list(rotate(list(rotate(M,90)), 90))

        M = np.array(M).flatten()
        M = M - np.mean(M)
        M = sklearn.preprocessing.normalize(np.array([M]), norm='l2', axis=1, copy=True, return_norm=False)[0]
        list_weights.append(M)

    Weights = np.array(list_weights)
    assert round(sum(Weights[0, :]) - 0, 3) == 0
    assert round(np.linalg.norm(Weights[0, :])- 1,3) == 0

    Weights = float(m) * float(STD) * Weights
    assert round(np.std(Weights[0, :])- STD,3) == 0
    assert round(np.std(Weights)- STD,3) == 0

    Weights=Weights.reshape(prim_circle_num, m,m)

    if thresh:
        for i, weight in enumerate(Weights):
            Weights[i][Weights[i]>=thresh*STD]=1
            Weights[i][Weights[i]<thresh*STD]=0

    return Weights


def train_topnet(net, optimizer, train_X, train_y, batchSize, test_X, test_y, imageDim, device,
                 batchesPerTest, loss_function, epochs=1, convGrids=None, freezeLayers=None):
    accuracies = []
    losses=[]
    for epoch in range(epochs):
        for n in range(0, len(train_X), batchSize):
            batch_X = train_X[n:n+batchSize].view(-1,1,imageDim[0],imageDim[1]).to(device)
            batch_y = train_y[n:n+batchSize].to(device)
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()

            # Keep convGrids frozen            
            if convGrids:
                with torch.no_grad():
                    for g, grids in enumerate(convGrids):
                        if np.any(grids) and not(freezeLayers[g]):
                            grids=torch.tensor(grids).to(device)
                            for x,param in enumerate(net.convnets[g].parameters()):
                                if(x==0):
                                    for i,grad in enumerate(param.grad):
                                        param.grad[i] = grad*grids[i]


            optimizer.step()
            if(n!=0 and n%(batchSize*batchesPerTest)==0 or n==len(train_X)):
                losses.append(loss)
                accuracies.append(test_topnet(net=net, X=test_X, y=test_y,
                                              batchSize=batchSize, imageDim=imageDim, device=device))
    return(losses,accuracies)

def test_topnet(net, X, y,batchSize, imageDim, device):
    correct  = 0
    total = 0
    with torch.no_grad():
        for n in range(0, len(X), batchSize):
            net_out = net(X[n:n+batchSize].view(-1,1,imageDim[0],imageDim[1]).to(device))
            for i, x in enumerate(net_out):
                if torch.argmax(y[n:n+batchSize][i])==torch.argmax(x):
                    correct +=1
                total +=1
    return(correct/total)

def build_config(config):
    ret = []
    item_cache=dict()
    for item_set in config:
        item_set_ret = []
        if item_set:
            for item in item_set:
                if item in item_cache.keys():
                    item_set_ret.append(item_cache[item])
                else:
                    if item[0]=='Circle':
                        new = primary_circle(item[1],item[2], thresh=item[3])
                    else:
                        if item[0]=='Klein':
                            new = legendre_klein_bottle(item[1],item[2], item[3], thresh=item[4])
                        else:
                            if item[0]=='Ones':
                                new = np.ones((item[1],item[2],item[2]))
                            else:
                                if item[0]=='Random':
                                    new = (np.random.rand(item[1],item[2],item[2])*2)-1 # Scale to -1<x<1
                                else: 
                                    if item[0]=='Gabor':
                                        new = small_gabor_set(item[1])
                                    else:
                                        raise('Must be "Circle" or "Klein" or "Ones" or "Random" or "Gabor"')
                    item_cache[item] = new
                    item_set_ret.append(new)
            # If we have appended filter schemas, flatten them into a single array
            if len(item_set_ret) > 1: item_set_ret = [filt for i_s in item_set_ret for filt in i_s]
            else: item_set_ret = item_set_ret[0]
            ret.append(np.array(item_set_ret))
        else:
            ret.append(None)
    return(ret) 

def run_configs(configs, results, returnNet=False):
    if not results:
        results = dict()
    for name, config in configs.items():
        weights_config = build_config(config['weights_config'])
        membership_config = build_config(config['membership_config'])

        net,optimizer = generate_network(
                                device=config['device'],
                                imageDim=config['imageDim'],
                                k=config['classes'], 
                                convLayerSpecs=config['convLayerSpecs'],
                                activations=config['activations'],
                                pools=config['pools'],
                                weightGrids=weights_config,
                                freezeLayers=config['freeze_layers'],
                                convGrids=membership_config,
                                lr=config['learning_rate'])

        loss,accuracies = train_topnet(
                     device=config['device'],
                     net=net, 
                     imageDim=config['imageDim'],
                     optimizer=optimizer, 
                     train_X=config['train_X'], 
                     train_y=config['train_y'], 
                     batchSize=config['batchSize'],
                     batchesPerTest=config['batchesPerTest'],
                     loss_function=get_topnet_func(config['loss_function'])(),
                     test_X=config['test_X'], 
                     test_y=config['test_y'],
                     freezeLayers=config['freeze_layers'],
                     convGrids=membership_config,
                     epochs=config['epochs'])
        results[name] = dict(loss=np.array(loss).astype(np.float32), accuracies=accuracies, config=config)
        if(returnNet): results[name]['net']=copy.deepcopy(net)
        del net, optimizer
    return(results)

def _plot_config(configuration, dataset, nLayers, itemName):
    for name, item in configuration.items():
        if len(item['config']['convLayerSpecs'])==nLayers or nLayers==None:
            plt.plot([i*item['config']['batchesPerTest'] for i in range(len(item['accuracies']))],
                     item[itemName], label=name)    
    plt.ylabel('Accuracy')
    plt.xlabel('Batches of '+ str(next(iter(configuration.items()))[1]['config']['batchSize']))
    plt.title(dataset)
    legend = plt.legend(loc='lower right', shadow=True, bbox_to_anchor=(1.61, 0.25))
    plt.show()

    # If separateByLayerCount argument is used, than we plot up to nLayers separately
def plot_results(results, dataset=None, nLayers=None, separateByLayerCount=False, item='accuracies'):
    if separateByLayerCount:
        for layerNumber in range(nLayers):
            layerNumber+=1
            if dataset:
                configuration=results[dataset]
                _plot_config(configuration, dataset, layerNumber, item)
            else:
                for dataset, configuration in results.items():
                    _plot_config(configuration, dataset, layerNumber, item)
    else:
            if dataset:
                configuration=results[dataset]
                _plot_config(configuration, dataset, nLayers, item)
            else:
                for dataset, configuration in results.items():
                    _plot_config(configuration, dataset, nLayers, item)
            
            
            
            
            
            
            
            
            
            