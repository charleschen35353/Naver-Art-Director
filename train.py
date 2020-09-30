import copy
import nsml
import utils
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from time import time, gmtime, strftime, localtime
import math

def max_gain(labels):

    discounts = torch.Tensor([np.log(2)/np.log(labels.shape[0] - x + 1) for x in range(labels.shape[0])]).cuda()
    
    return (discounts*(2**labels.sort()[0] -1 )).sum()

def train_test(model, optimizer, scheduler,dataloaders, dataset_sizes, args):

    start_time = time()
    print("[t] train start time : ", strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6
    best_loss_at = 0
    best_acc = 0.0
    best_acc_at = 0
    train_best_acc = 0.0
    train_best_acc_at = 0
    train_best_loss = 1e6
    train_best_loss_at = 0
    
    with torch.no_grad():
        rank = dict(train={}, test={})
        for phase in ['train', 'test']:
            ctrs, adids = [], []
            for i, data in enumerate(dataloaders[phase]):
                _, _, ctr, _, _, adid = data
                ctrs.append(ctr.detach().numpy())
                adids.append(adid.detach().numpy())
            # [N, a] + [M, a] -> [N + M, a]
            ctrs = np.concatenate(ctrs, axis=0).reshape(-1)
            adids = np.concatenate(adids, axis=0).reshape(-1)
            adids = adids[ctrs.argsort()[::-1]].reshape(-1)
            for i in range(adids.shape[0]):
                rank[phase][adids[i]] = i

    for epoch in range(args.epochs):
        print('[t] Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                prev_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                model.train(False)  # Set model to evaluate mode

            running_loss, running_corrects = 0.0, 0
            epoch_results, epoch_start = [], time()
            epoch_probs, epoch_ctrs, epoch_ids, mes = [], [], [], []

            whole_inputs = []
            whole_ctrs = []
            whole_labels = []
            whole_outputs = []
            result_ctrs = None
            ch, h ,w = None, None, None
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                itr_start = time()
                outputs = None
                inputs, labels, ctrs, sexs, age_areas, adids = data
                
                # wrap them in Variable
                if args.use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.float().cuda())
                    ctrs = Variable(ctrs.float().cuda())
                    sexs = Variable(sexs.float().cuda())
                    age_areas = Variable(age_areas.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    ctrs, sexs, age_areas = Variable(ctrs.float()), Variable(sexs.float()), Variable(age_areas.float())
                
                _,ch,h,w = inputs.shape
                

                if args.use_multicrop and phase == 'test':
                    bs, ncrops, ch, h, w = inputs.size()
                    outputs = model(inputs.view(-1, ch, h, w), sexs, age_areas)
                    outputs = outputs.view(bs, ncrops, -1).mean(1)
                else:
                    outputs = model(inputs, sexs, age_areas)

                if phase == 'test':
                    whole_inputs.append(inputs)
                    whole_outputs.append(outputs)
                    whole_ctrs.append(ctrs)
                    whole_labels.append(labels)
                    
                order = outputs.sort()[1][-1]
                epoch_probs.append(outputs.detach())
                epoch_ctrs.append(ctrs.detach())
                epoch_ids.append(adids.detach())
                mes.append(ctrs.detach()[order])

                optimizer.zero_grad()
                if args.method == 'regression':
                    
                    # forward
                    #loss = F.mse_loss(outputs.squeeze(), ctrs.squeeze())

                    ###soft rank loss softndcg###
                    if args.softNDCG_sigma > 0:   
                        if args.softNDCG_at == -1:
                            args.softNDCG_at = args.batch_size
                        ELEMENT_COUNT = ctrs.shape[0]
                        #probabilities of outputs: Si~N(outputs[i],sigma)
                        #probability that Si > SJ: returned by function prij(output[i],output[j],sigma) 
                        #construct a relative probablity table: enrty[i][j] denotes the probability
                        #that doci is place at the higher ranking than doc j
                        
                        #rel_prob = torch.zeros([ELEMENT_COUNT,ELEMENT_COUNT]).cuda()
                        rel_prob = 1 - 0.5 * (1 + torch.erf( (-outputs+outputs.transpose(0,1)) / 2**0.5 / (2*args.softNDCG_sigma)**0.5 ))


                        #takes shape [i][r],the probability that ith element is place at rth position 
                        #construct the table
                        ################## This takes O(n^3) need to be opitmized##################
                        position_probs = []
                        for itr in range(ELEMENT_COUNT-1): #iteration count
                            position_prob = torch.zeros([ ELEMENT_COUNT, ELEMENT_COUNT]).cuda()
                            if itr == 0:
                                position_prob[:,0] = 1
                                position_probs.append(position_prob)
                                position_probs.append(torch.zeros([ ELEMENT_COUNT, ELEMENT_COUNT]).cuda())
                            else:
                                position_probs.append(position_prob)

                            for j in range(ELEMENT_COUNT): #elementwise
                                for r in range(ELEMENT_COUNT):
                                    new_add_ind = itr
                                    if itr >= j:
                                        new_add_ind += 1
                                    if r-1 < 0:
                                        position_probs[itr+1][j][r] = position_probs[itr][j][r]*(1-rel_prob[new_add_ind][j])
                                    else:
                                        position_probs[itr+1][j][r] = position_probs[itr][j][r-1]*rel_prob[new_add_ind][j] +  position_probs[itr][j][r]*(1-rel_prob[new_add_ind][j])

                        #expectation value of ranking of the ith element
                        discounted_factors = torch.Tensor([np.log(2)/np.log(x+2) for x in range(ELEMENT_COUNT)]).cuda()
                        #vectorized softNDGC
                        
                        #print(position_prob[ELEMENT_COUNT-1].mm(discounted_factors.view(ELEMENT_COUNT,1))) #expected discount
                        softNDCG = ((2**labels - 1)*(position_probs[ELEMENT_COUNT-1].mm(discounted_factors.view(ELEMENT_COUNT,1))).view(ELEMENT_COUNT)) / max_gain(labels)
                        softNDCG_at = softNDCG[:args.softNDCG_at]
                        loss = 1 - softNDCG_at.sum()
                        if math.isnan(loss):
                            loss = (outputs * torch.Tensor([0]).cuda()).sum()

                    if  i % 15 ==0 and not args.whole_batch:
                        print('[{}] [{:d}/{:d}]iter, Loss: {:.4f}'
                          .format(phase,i,
                                  dataset_sizes[phase]//args.batch_size,
                                  loss.item()))
                
                elif args.method == 'ce_mse':

                    loss_ce = F.cross_entropy(outputs, labels)

                    _, preds = torch.max(outputs.data, 1)

                    running_corrects += torch.sum(preds == labels.data).item()

                    if args.use_gpu:
                        preds = Variable(preds.float().cuda())
                        labels_for_loss = labels.float().cuda()
                    else:
                        preds = Variable(preds.float())
                        labels_for_loss = labels.float()

                    loss_mse = F.mse_loss(preds.squeeze(), labels_for_loss.squeeze())
                    loss = loss_ce + loss_mse

                    print('[{:d}/{:d}]iter, Loss: {:.4f}, ''CE : {:.4f}, MSE : {:.4f}'
                          .format(i,
                                  dataset_sizes[phase] // args.batch_size,
                                  loss.item(), loss_ce.item(), loss_mse.item()),end=" ")

                elif args.method == 'classification':
                    loss = F.cross_entropy(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    running_corrects += torch.sum(preds == labels.data).item()

                    if  i % 15 ==0:
                        print('[{}] [{:d}/{:d}]iter, Loss: {:.4f}'
                          .format(phase,i,
                                  dataset_sizes[phase]//args.batch_size,
                                  loss.item()))
                elif args.method == 'relu_clipping':
                    loss = F.binary_cross_entropy(outputs, labels)

                else:
                    raise AssertionError()

                running_loss += loss.item() * inputs.size(0)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward(retain_graph = True)
                    optimizer.step()

                if epoch % 10 == 0:
                    if args.method in ['ce_mse', 'classification'] :
                        a, b = preds, labels
                    elif args.method in ['regression']:
                        a, b = outputs, ctrs
                    a = a.cpu().squeeze().tolist()
                    b = b.cpu().squeeze().tolist()
                    # torch.Tensor([0.1]).squeeze().tolist() is float, not list
                    if not isinstance(a, list):
                        a, b = [a], [b]
                    epoch_results.extend(list(zip(list(a),list(b))))
            
            # Cross comparison
            if phase == 'test':
                #iterate until successfully reduce data to a certain amount 
                while args.batch_size * len(whole_outputs) >= 128: 
                    #append to lists whose size is small than batch_size
                    NULL_INPUT = torch.zeros(1,ch,h,w).cuda()
                    NULL_OUTPUT = torch.zeros(1,1).cuda()
                    NULL_CTR = torch.zeros(1).cuda()

                    #track inputs of the output
                    i1 = whole_inputs.pop()
                    i2 = whole_inputs.pop()
                    while i1.shape[0] < args.batch_size:
                        i1 = torch.cat((i1,NULL_INPUT))

                    #track outputs
                    l1 = whole_outputs.pop()
                    while l1.shape[0] < args.batch_size:
                        l1 = torch.cat((l1,NULL_OUTPUT))
                    l1_sort = l1.view(-1).sort(descending = True)
                    l2 = whole_outputs.pop()
                    l2_sort = l2.view(-1).sort(descending = True)
                    
                    #track ctrs of the outputs
                    combined_ctr = 0 
                    ctr1 = whole_ctrs.pop()
                    ctr2 = whole_ctrs.pop()
                    while ctr1.shape[0] < args.batch_size:
                        ctr1 = torch.cat((ctr1,NULL_CTR))
                    
                    #calculate combined ctrs
                    for e in l1_sort[1][:args.batch_size//2]:
                        combined_ctr += ctr1[e]
                        #print(combined_ctr)
                    for e in l2_sort[1][:args.batch_size//2]:
                        combined_ctr += ctr2[e]
                        #print(combined_ctr)
                    #print(combined_ctr/args.batch_size)
                    #print(ctr1.mean())
                    #print(ctr2.mean())

                    #extract first half
                    if combined_ctr/args.batch_size > ctr1.mean() and combined_ctr/args.batch_size> ctr2.mean():
                        #print("AAAAAAAAAAAAAAA")
                        new_input = None
                        #ch, h, w = None,None,None
                        for ind in l1_sort[1][:args.batch_size//2]:
                            if new_input is None:
                                new_input = i1[ind]
                            #    ch,h,w = new_input.shape
                            else:
                                new_input = torch.cat( (new_input,i1[ind]) )
                        for ind in l2_sort[1][:args.batch_size//2]:  
                            new_input = torch.cat( (new_input, i2[ind]) )#.view(-1,ch,h,w)
                        new_input = new_input.view(-1,ch,h,w)
                        #print(new_input.shape)
                        new_output = model(new_input)
                        whole_outputs.insert(0, new_output)
                        whole_inputs.insert(0,new_input)
                        #whole_outputs.insert(0, torch.cat( (l1_sort[0][:batch_size//2],l2_sort[0][:batch_size//2]) ) )
                        
                        
                        new_ctr = torch.Tensor([]).cuda()           
                        for c in l1_sort[1][:args.batch_size//2]:
                            new_ctr = torch.cat( (new_ctr, ctr1[c].view(-1) ) )
                        for c in l2_sort[1][:args.batch_size//2]:
                            new_ctr = torch.cat( (new_ctr, ctr2[c].view(-1) ) )
                        whole_ctrs.insert(0, new_ctr)
                    
                    #level 2 comparison
                    else:
                        combined_ctr = [0,0]
                        
                        for e in l1_sort[1][: args.batch_size//4*3]:
                            combined_ctr[0] += ctr1[e]
                            combined_ctr[1] += ctr2[e]
                        for e in l2_sort[1][:args.batch_size//4]:
                            combined_ctr[0] += ctr2[e]
                            combined_ctr[1] += ctr1[e]
                        
                        #susccessful casess for level 2 comaprison: 0.75:0.25
                        if combined_ctr[0]/args.batch_size > ctr1.mean() and combined_ctr[0]/args.batch_size > ctr2.mean():
                            #print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
                            new_input = None
                            #ch, h, w = None, None, None
                            for ind in l1_sort[1][:args.batch_size//4*3]:
                                if new_input is None:
                                    new_input = i1[ind]
                                    #ch,h,w = new_input.shape
                                else:
                                    new_input = torch.cat( (new_input,i1[ind]) )#.view(-1,ch ,h,w)
                            for ind in l2_sort[1][:args.batch_size//4]:  
                                new_input = torch.cat( (new_input,i2[ind]) )#.view(-1,ch,h,w)
                            new_input = new_input.view(-1,ch,h,w)
                            new_output = model(new_input)
                            whole_outputs.insert(0, new_output)
                            whole_inputs.insert(0,new_input)
                            #whole_outputs.insert(0, torch.cat( (l1_sort[0][:batch_size//2],l2_sort[0][:batch_size//2]) ) )
                            
                            new_ctr = torch.Tensor([]).cuda()
                                                        
                            for c in l1_sort[1][:args.batch_size//4*3]:
                                new_ctr = torch.cat( (new_ctr, ctr1[c].view(-1) ) )
                            for c in l2_sort[1][:args.batch_size//4]:
                                new_ctr = torch.cat( (new_ctr, ctr2[c].view(-1) ) )
                            whole_ctrs.insert(0, new_ctr)
                        
                        #susccessful casess for level 2 comaprison: 0.25:0.75
                        elif combined_ctr[1]/args.batch_size > ctr1.mean() and combined_ctr[1]/args.batch_size> ctr2.mean():
                            #print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
                            new_input = None
                            #ch,h,w = None, None,None
                            for ind in l1_sort[1][:args.batch_size//4]:
                                if new_input is None:
                                    new_input = i1[ind]
                                #    ch,h,w = new_input.shape
                                else:
                                    new_input = torch.cat( (new_input,i1[ind]) )#.view(-1,ch ,h,w)
                            for ind in l2_sort[1][:args.batch_size//4*3]:  
                                new_input = torch.cat( (new_input,i2[ind]) )#.view(-1,ch,h,w)
                            new_input = new_input.view(-1,ch,h,w)
                            new_output = model(new_input)
                            whole_outputs.insert(0, new_output)
                            whole_inputs.insert(0,new_input)
                            #whole_outputs.insert(0, torch.cat( (l1_sort[0][:batch_size//2],l2_sort[0][:batch_size//2]) ) )
                            
                            new_ctr = torch.Tensor([]).cuda()

                            for c in l1_sort[1][:args.batch_size//4]:
                                new_ctr = torch.cat( (new_ctr, ctr1[c].view(-1) ) )
                            for c in l2_sort[1][:args.batch_size//4*3]:
                                new_ctr = torch.cat( (new_ctr, ctr2[c].view(-1) ) )
                            whole_ctrs.insert(0, new_ctr)

                        #cros comparison doesnt find better combination performance. Add back original data
                        else:
                            #print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                            if ctr1.mean() > ctr2.mean():
                                whole_outputs.insert(0,l1)
                                whole_ctrs.insert(0,ctr1)
                                whole_inputs.insert(0,i1)
                            else:
                                whole_outputs.insert(0,l2)
                                whole_ctrs.insert(0,ctr2)
                                whole_inputs.insert(0,i2)
                    '''
                    print(ctr1)
                    print(ctr2)
                    print(whole_ctrs[0])
                    print('========================')
                    '''
                #whole_outputs and whole_ctrs
                top_outputs = torch.Tensor([])
                top_ctrs = torch.Tensor([])

                
                for e in range(len(whole_outputs)):
                    top_outputs = torch.cat((top_outputs, whole_outputs[e].cpu()))
                    top_ctrs = torch.cat((top_ctrs, whole_ctrs[e].cpu()))    
                '''
                print("===================")
                print(top_outputs)
                print(top_ctrs)
                '''
                result_ctrs = torch.Tensor([])
                for c in top_outputs.view(-1).sort(descending = True)[1]:
                    result_ctrs = torch.cat( (result_ctrs, top_ctrs[c].view(-1)) )
                #print(result_ctrs)


            epoch_loss = running_loss / dataset_sizes[phase]

            # Calculate top-k avg ctr
            if args.method == 'regression':
                epoch_probs = torch.cat(epoch_probs, 0).cpu().detach().numpy().reshape(-1, 1)
            else:
                epoch_probs = torch.cat(epoch_probs, 0).cpu().detach().numpy()[:,1].reshape(-1, 1)
            epoch_ctrs = torch.cat(epoch_ctrs, 0).cpu().detach().numpy().reshape(-1, 1)
            epoch_ids = torch.cat(epoch_ids, 0).cpu().detach().numpy().reshape(-1, 1)

            # [N, 1], [N, 1] -> [N, 2]
            concat = np.concatenate((epoch_probs, epoch_ctrs, epoch_ids), axis=1)
            concat = concat[concat[:,0].argsort()][::-1]
            l1_diff = np.array([
                abs(rank[phase][concat[j][2]] - j) for j in range(50)
            ]).reshape(-1).astype(np.float32)
            concat = concat[:,1].reshape(-1)
            
            

            top_k_ctrs = [
                float(np.mean(concat[:10])), 
                float(np.mean(concat[:25])), 
                float(np.mean(concat[:50]))
            ]
            print("A: {}".format(float(np.mean(concat[:50]))))
            
            if args.softNDCG_sigma > 0 and phase == 'test':
                result_ctrs = result_ctrs.numpy()
                top_k_ctrs = [
                    max( float(np.mean(concat[:10])), float(np.mean(result_ctrs[:10]))), 
                    max( float(np.mean(concat[:25])), float(np.mean(result_ctrs[:25]))), 
                    max( float(np.mean(concat[:50])), float(np.mean(result_ctrs[:50]))), 
                ]
            if phase == 'test':
                print("B: {}".format(float(np.mean(result_ctrs[:50]))))
            rank_MD = [
                float(np.sum(l1_diff[:10])),
                float(np.sum(l1_diff[:25])),
                float(np.sum(l1_diff[:50]))
            ]
            l2_diff = np.square(l1_diff)
            rank_ED = [
                float(np.sqrt(np.sum(l2_diff[:10]))),
                float(np.sqrt(np.sum(l2_diff[:25]))),
                float(np.sqrt(np.sum(l2_diff[:50])))
            ]

            nsml_kwargs = dict(
                top_10_ctr=top_k_ctrs[0],
                top_25_ctr=top_k_ctrs[1],
                top_50_ctr=top_k_ctrs[2],
                top_10_rank_MD=rank_MD[0],
                top_25_rank_MD=rank_MD[1],
                top_50_rank_MD=rank_MD[2],
                top_10_rank_ED=rank_ED[0],
                top_25_rank_ED=rank_ED[1],
                top_50_rank_ED=rank_ED[2],
            #    NDCG = NDCG
            )
            nsml_kwargs = {
                phase + '_' + k : nsml_kwargs[k] for k in nsml_kwargs
            }

            if args.method == 'classification' or args.method == 'ce_mse':
                epoch_acc = running_corrects / dataset_sizes[phase]
                if phase == 'train':
                    if args.lr_sch in [1, 2, 3, 4, 5]:
                        scheduler.step()
                    else:
                        scheduler.step(epoch_loss)
                    nsml.report(
                        step=epoch,
                        epoch_total=args.epochs + 1,
                        train_accuracy=epoch_acc,
                        train_loss=epoch_loss,
                        **nsml_kwargs
                    )
                    if epoch_acc > train_best_acc:
                        train_best_acc = epoch_acc
                        train_best_acc_at = epoch
                    print('[{}] Loss: {:.4f} Acc: {:.4f} time: {:f}s current train Best : {:4f} at {:d}'.format(phase, epoch_loss, epoch_acc,time()-epoch_start,train_best_acc,train_best_acc_at))

                elif phase == 'test':
                    nsml.report(
                        step=epoch,
                        epoch_total=args.epochs + 1,
                        epoch_now=epoch,
                        test_loss=epoch_loss,
                        test_accuracy=epoch_acc,
                        summary=True,
                        **nsml_kwargs
                    )

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_acc_at = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        nsml.save(epoch)

                    #print('[train] New Best Acc: {:4f} at {:d}'.format(best_acc,best_acc_at))
                    print('[{}] Loss: {:.4f} Acc: {:.4f} time: {:f}s current test Best : {:4f} at {:d}'.format(phase, epoch_loss, epoch_acc,time()-epoch_start,best_acc,best_acc_at))
            else:

                if phase == 'train':
                    if args.lr_sch in [1, 2, 3, 4, 5]:
                        scheduler.step()
                    else:
                        scheduler.step(epoch_loss)
                    nsml.report(
                        step=epoch,
                        epoch_total=args.epochs + 1,
                        train_loss=epoch_loss,
                        **nsml_kwargs
                    )
                    if epoch_loss < train_best_loss: 
                        train_best_loss = epoch_loss
                        train_best_loss_at = epoch
                    print('[{}] Loss: {:.4f} time: {:f}s current Best : {:4f} at {:d}'.format(phase, epoch_loss ,time()-epoch_start,train_best_loss,train_best_loss_at))
                elif phase == 'test':
                    nsml.report(
                        step=epoch,
                        epoch_now=epoch,
                        epoch_total=args.epochs + 1,
                        test_loss=epoch_loss,
                        summary=True,
                        **nsml_kwargs
                    )

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_loss_at = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        nsml.save(epoch)
                    print('[{}] Loss: {:.4f} time: {:f}s current Best : {:4f} at {:d}'.format(phase, epoch_loss ,time()-epoch_start,best_loss,best_loss_at))
            if phase == 'test':
                torch.set_grad_enabled(prev_grad_state) 



    time_elapsed = time() - start_time
    print('[t] Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if args.method == 'regression':
        print('[t] Best test Loss: {:4f}\nat\n{:d}'.format(best_loss,best_loss_at))
        nsml.report(
            step=args.epochs,
            best_loss=best_loss,
            best_loss_at=best_loss_at,
            summary=True
        )
    else:
        print('[t] Best val Acc: {:4f}\nat\n{:d}'.format(best_acc,best_acc_at))
        nsml.report(
            step=args.epochs,
            best_acc=best_acc,
            best_acc_at=best_acc_at,
            summary=True
        )
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
