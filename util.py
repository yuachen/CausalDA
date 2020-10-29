import numpy as np
import sem

import torch

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def MSE(yhat, y):
    return np.mean((yhat-y)**2)

def torchMSE(a, b):
    return torch.mean((a.squeeze() - b)**2)

def torchloaderMSE(me, dataloader, device):
    # get MSE from a torch model with dataloader
    error = 0
    n = 0
    with torch.no_grad():
        for data in dataloader:
            x, y = data[0].to(device), data[1].to(device)
            ypred = me.predict(x)
            n += x.shape[0]
            error += torch.sum((ypred.squeeze() - y)**2)
    return error.item()/n


# given a SEM, run all methods and return target risks and target test risks
def run_all_methods(sem1, methods, n=1000, repeats=10, returnMinDiffIndx=False, tag_DA='DAMMD'):
    M = sem1.M
    results_src_all = np.zeros((M-1, len(methods), 2, repeats))
    results_tar_all = np.zeros((len(methods), 2, repeats))
    results_minDiffIndx = {}
    
    # generate data 
    for repeat in range(repeats):
        data = sem1.generateAllSamples(n)
        dataTest = sem1.generateAllSamples(n)
        # may use other target as well
        source = np.arange(M-1)
        target = M-1
        
        # prepare torch format data
        dataTorch = {}
        dataTorchTest = {}
          
        for i in range(M):
            dataTorch[i] = [torch.from_numpy(data[i][0].astype(np.float32)).to(device), 
                        torch.from_numpy(data[i][1].astype(np.float32)).to(device)]
            dataTorchTest[i] = [torch.from_numpy(dataTest[i][0].astype(np.float32)).to(device), 
                        torch.from_numpy(dataTest[i][1].astype(np.float32)).to(device)]
            
        # prepare torch format data for batch stochastic gradient descent
        train_batch_size = 500
        test_batch_size = 500

        trainloaders = {}
        testloaders = {}

        for i in range(M):
            train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data[i][0]),
                                           torch.Tensor(data[i][1]))
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(dataTest[i][0]),
                                           torch.Tensor(dataTest[i][1]))
            trainloaders[i] = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
            testloaders[i] = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

        for i, m in enumerate(methods):
            if m.__module__ == 'semiclass':
                me = m.fit(data, source=source, target=target)
                if hasattr(me, 'minDiffIndx'):
                    print("best index="+str(me.minDiffIndx))
                    results_minDiffIndx[(tag_DA, i, repeat)] = me.minDiffIndx
                xtar, ytar = data[target]
                xtar_test, ytar_test = dataTest[target]
                targetE = MSE(me.ypred, ytar)
                targetNE = MSE(me.predict(xtar_test), ytar_test)
                for j, sourcej in enumerate(source):
                    results_src_all[j, i, 0, repeat] = MSE(me.predict(data[sourcej][0]), data[sourcej][1])
                    results_src_all[j, i, 1, repeat] = MSE(me.predict(dataTest[sourcej][0]), dataTest[sourcej][1])
            elif  m.__module__ == 'semitorchclass':
                me = m.fit(dataTorch, source=source, target=target)
                if hasattr(me, 'minDiffIndx'):
                    print("best index="+str(me.minDiffIndx))
                    results_minDiffIndx[(tag_DA, i, repeat)] = me.minDiffIndx
                xtar, ytar= dataTorch[target]
                xtar_test, ytar_test= dataTorchTest[target]
                targetE = torchMSE(me.ypred, ytar)
                targetNE = torchMSE(me.predict(xtar_test), ytar_test)
                for j, sourcej in enumerate(source):
                    results_src_all[j, i, 0, repeat] = torchMSE(me.predict(dataTorch[sourcej][0]), dataTorch[sourcej][1])
                    results_src_all[j, i, 1, repeat] = torchMSE(me.predict(dataTorchTest[sourcej][0]), dataTorchTest[sourcej][1])
            elif m.__module__ == 'semitorchstocclass':
                me = m.fit(trainloaders, source=source, target=target)
                targetE = torchloaderMSE(me, trainloaders[target], device)
                targetNE = torchloaderMSE(me, testloaders[target], device)
                for j, sourcej in enumerate(source):
                    results_src_all[j, i, 0, repeat] = torchloaderMSE(me, trainloaders[sourcej], device)
                    results_src_all[j, i, 1, repeat] = torchloaderMSE(me, testloaders[sourcej], device)
            else:
                raise ValueError("Unexpected method class")
            results_tar_all[i, 0, repeat] = targetE
            results_tar_all[i, 1, repeat] = targetNE
            print("Repeat %d Target %-30s error=%.3f errorTest=%.3f" %(repeat, str(m), targetE, targetNE), flush=True)
    if returnMinDiffIndx:
        return results_src_all, results_tar_all, results_minDiffIndx
    else:
        return results_src_all, results_tar_all
