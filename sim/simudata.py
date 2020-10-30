import numpy as np
import sys
sys.path.append('../')
import sem
import myrandom

def pick_intervention_and_noise(M, dp1, inter2noise_ratio, interY=0., cic=[], typeshift='sm1', varAs=None, varnoiseY=1.):
    if typeshift == 'sm1':
        meanAs = inter2noise_ratio * np.random.randn(M, dp1)

#         meanAs[:, -1] = interY * np.random.randn(M) # 0 means no intervention on Y
        meanAs[:, -1] = 0.
        meanAs[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y

        meanAs[:, cic] = 0 # set conditional invariant components

        if not varAs:
            varAs = np.zeros((M, dp1))
        interAf = myrandom.Gaussian(M, meanAs, varAs)

        noisevar = np.ones((1, dp1))
        noisevar[-1] = varnoiseY
        noisef = myrandom.Gaussian(1, np.zeros((1, dp1)), noisevar)
    elif typeshift == 'sm2':
        # mixture
        meanAs1 = inter2noise_ratio * np.random.randn(M, dp1)
        meanAs2 = inter2noise_ratio * np.random.randn(M, dp1)
        meanAs1[:, -1] = 0. # 0 intervY only for the target env
        meanAs2[:, -1] = 0. # 0 intervY only for the target env
        meanAs1[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs2[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs1[:, cic] = 0 # set conditional invariant components
        meanAs2[:, cic] = 0 # set conditional invariant components
        meanAsList = [meanAs1, meanAs2]

        if not varAs:
            varAs = np.zeros((M, dp1))
        interAf =  myrandom.Mix2Gaussian(M, meanAsList, varAs)

        noisevar = np.ones((1, dp1))
        noisevar[-1] = varnoiseY
        noisef = myrandom.Gaussian(1, np.zeros((1, dp1)), noisevar)
    elif typeshift == 'sm3':
        # mixture
        meanAs1 = inter2noise_ratio * np.random.randn(M, dp1)
        meanAs2 = inter2noise_ratio * np.random.randn(M, dp1)
        meanAs3 = inter2noise_ratio * np.random.randn(M, dp1)
        meanAs1[:, -1] = 0. # 0 intervY only for the target env
        meanAs2[:, -1] = 0. # 0 intervY only for the target env
        meanAs3[:, -1] = 0. # 0 intervY only for the target env
        meanAs1[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs2[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs3[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs1[:, cic] = 0 # set conditional invariant components
        meanAs2[:, cic] = 0 # set conditional invariant components
        meanAs3[:, cic] = 0 # set conditional invariant components
        meanAsList = [meanAs1, meanAs2, meanAs3]

        if not varAs:
            varAs = np.zeros((M, dp1))
        interAf =  myrandom.MixkGaussian(M, meanAsList, varAs)

        noisevar = np.ones((1, dp1))
        noisevar[-1] = varnoiseY
        noisef = myrandom.Gaussian(1, np.zeros((1, dp1)), noisevar)

    elif typeshift == 'sv1':
        meanAs = np.zeros((M, dp1))
        varAs_inter = inter2noise_ratio * (np.abs(np.random.randn(M, dp1)))
        varAs_inter[:, -1] = 1.
        varAs = varAs_inter

        interAf = myrandom.Gaussian(M, meanAs, varAs)

        # the noise is always zero, is taken care of by interAf
        noisef = myrandom.Gaussian(1, np.zeros((1, dp1)), np.zeros((1, dp1)))

    elif typeshift == 'smv1':
        meanAs = inter2noise_ratio * np.random.randn(M, dp1)
        # meanAs[:, -1] = interY * np.random.randn(M) # 0 means no intervention on Y
        meanAs[:, -1] = 0.
        meanAs[-1, -1] = interY * np.random.randn(1) # 0 means no intervention on Y
        meanAs[:, cic] = 0 # set conditional invariant components
        varAs_inter = inter2noise_ratio * np.abs(np.random.randn(M, dp1))
        varAs_inter[:, -1] = 1.
        varAs_inter[:, cic] = 1. # set conditional invariant components
        varAs = varAs_inter

        interAf = myrandom.Gaussian(M, meanAs, varAs)

        # the noise is always zero, is taken care of by interAf
        noisef = myrandom.Gaussian(1, np.zeros((1, dp1)), np.zeros((1, dp1)))

    return interAf, noisef

def pick_random_B(pred_dir = 'anticausal', dp1=1):
    B = np.zeros((dp1, dp1))
    if pred_dir == 'anticausal':
        # triangular B
        for i in range(0, dp1-1):
            for j in range(i+1, dp1-1):
                B[j, i] = 0.5 * np.random.randn(1)
        B[:, -1] = 1.0 * np.random.randn(dp1)
        B[-1, -1] = 0
    elif pred_dir == 'causal':
        for i in range(0, dp1-1):
            for j in range(i+1, dp1-1):
                B[j, i] = 0.5 * np.random.randn(1)
        # so y should not change X
        B[:, -1] = 0
        # causal prediction
        B[-1, :-1] = 1.0 * np.random.randn(dp1-1)
        B[-1, -1] = 0
    elif pred_dir == 'halfhalf':
        # make them mixed causal and anti-causal
        B = np.zeros((dp1, dp1))
        for i in range(0, dp1-1):
            for j in range(i+1, dp1-1):
                B[j, i] = 0.5 * np.random.randn(1)
        for i in range((dp1-1)//2):
            # half anti-causal
            B[2*i, -1] = 1.0 * np.random.randn(1)
            # half causal
            if 2*i+1 <= dp1-2:
                B[-1, 2*i+1] = 1.0 * np.random.randn(1)
                for j in range((dp1-1)//2):
                    # anti-causal node should not point to causal node, to ensure acyclic graph
                    B[2*i+1, 2*j] = 0
        B[-1, -1] = 0
    else:
        raise ValueError('case not recognized.')


    return B


def pick_sem(data_num, params = None, seed=123456):
    np.random.seed(seed)
    # name rules
    # r0: r0 regression Y is cause
    #     r1 regression Y is effect
    #     r2 regression Y is in the middle
    # sm1: 1 dimensional mean shift
    #     case 1 1 dimensional mean shift
    #     case 2 2 mixture of mean shift
    #     sv1 change of variance
    # d3: dimension of the problem is 3
    # x1: no intervention on Y, only intervention on X,
    #     case 1 (no conditional invariant components, no inter Y)
    #     case 2 (with conditional invariant components, no inter Y)
    #     case 3 (no conditional invariant components, inter Y)
    #     case 4 (with conditional invariant components, inter Y)
    if 'd3' in data_num:
        # Y cause, d = 3
        M = params['M']
        # d plus 1
        dp1 = 4


        inter2noise_ratio = params['inter2noise_ratio']
        if 'x1' in data_num:
            # conditional invariant components
            cic = []
            # intervention on Y
            interY = 0
        elif 'x3' in data_num:
            cic = []
            if 'interY' in params.keys():
                interY = params['interY']
            else:
                interY = 1.
        elif 'x2' in data_num:
            # conditional invariant components
            cic = [0]
            # intervention on Y
            interY = 0
        elif 'x4' in data_num:
            cic = [0]
            if 'interY' in params.keys():
                interY = params['interY']
            else:
                interY = 1.
        else:
            raise ValueError('case not recognized.')

        if 'sm1' in data_num:
            typeshift = 'sm1'
        elif 'sm2' in data_num:
            typeshift = 'sm2'
        elif 'sm3' in data_num:
            typeshift = 'sm3'
        elif 'sv1' in data_num:
            typeshift = 'sv1'
        elif 'smv1' in data_num:
            typeshift = 'smv1'
        elif 'smm2' in data_num:
            typeshift = 'smm2'
        else:
            typeshift = 'sm1'


        interAf, noisef = pick_intervention_and_noise(M, dp1, inter2noise_ratio, interY=interY,
                                                      cic=cic, typeshift=typeshift, varAs=None, varnoiseY=1.)

        if 'r0' in data_num:
            pred_dir = 'anticausal'
            B = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 3], [0, 0, 0, 0]])
        elif 'r1' in data_num:
            pred_dir = 'causal'
            B = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 3, 0]])
#         elif 'r2' in data_num:
#             pred_dir = 'halfhalf'
        else:
            raise ValueError('case not recognized.')



        invariantList = []
        message = "%sM%di%d, fixed simple B" %(data_num, M, inter2noise_ratio)
    elif 'd?' in data_num:
        # Y cause, d = ?
        M = params['M']
        # d plus 1
        if 'd' in params.keys():
            dp1 = params['d'] + 1
        else:
            # set default dimension to 10
            dp1 = 11
        if 'interY' in params.keys():
            interY = params['interY']
        else:
            interY = 1.

        inter2noise_ratio = params['inter2noise_ratio']

        if 'r0' in data_num:
            pred_dir = 'anticausal'
            varnoiseY = 1.
        elif 'r1' in data_num:
            pred_dir = 'causal'
            varnoiseY = 1.
        elif 'r2' in data_num:
            pred_dir = 'halfhalf'
            varnoiseY = 0.01
        else:
            raise ValueError('case not recognized.')

        B = pick_random_B(pred_dir, dp1)

        if 'x1' in data_num:
            # conditional invariant components
            cic = []
            # intervention on Y
            interY = 0
        elif 'x3' in data_num:
            cic = []
            if 'interY' in params.keys():
                interY = params['interY']
            else:
                interY = 1.
        elif 'x2' in data_num:
            if 'cicnum' in params.keys():
                cicnum = params['cicnum']
            else:
                cicnum = int(dp1/2)
            # conditional invariant components
            cic = np.random.choice(dp1-1, cicnum, replace=False)
            # intervention on Y
            interY = 0
        elif 'x4' in data_num:
            if 'cicnum' in params.keys():
                cicnum = params['cicnum']
            else:
                cicnum = int(dp1/2)
#             cic = np.arange(0, cicnum)
            cic = np.random.choice(dp1-1, cicnum, replace=False)
            if 'interY' in params.keys():
                interY = params['interY']
            else:
                interY = 1.
        else:
            raise ValueError('case not recognized.')

        if 'sm1' in data_num:
            typeshift = 'sm1'
        elif 'sm2' in data_num:
            typeshift = 'sm2'
        elif 'sv1' in data_num:
            typeshift = 'sv1'
        elif 'smv1' in data_num:
            typeshift = 'smv1'
        elif 'smm2' in data_num:
            typeshift = 'smm2'
        else:
            typeshift = 'sm1'

        interAf, noisef = pick_intervention_and_noise(M, dp1, inter2noise_ratio, interY=interY, cic=cic, typeshift=typeshift, varAs=None, varnoiseY=varnoiseY)

        invariantList = cic
        message = "%sM%dd%di%d, fixed simple B" %(data_num, M, dp1-1, inter2noise_ratio)
    # generate sem
    sem1 = sem.SEM(B, noisef, interAf, invariantList=invariantList, message=message)

    return sem1
