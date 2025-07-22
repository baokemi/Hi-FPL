import argparse
from pathlib import Path
import numpy as np
import torch
import random
import os
from utils import *
from office_dataset import prepare_data
from domainnet_dataset import prepare_data_domain
import json
from models.vit import CONFIGS,VisionTransformer_prompt, RepCatcher
from models.vit import *
import sys
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import MultiheadAttention
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='ViT-B_16', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--client_lr', type=float, default=0.1, help='learning rate (default: 0.01)')
    parser.add_argument('--corr_lr', type=float, default=0.00001, help='learning rate (default: 0.01)')
    parser.add_argument('--dac_lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_parties', type=int, default=100,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='HiFPL',  help='strategy')
    parser.add_argument('--comm_round', type=int, default=120, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.05, help='Sample ratio for each communication round')
    parser.add_argument('--test_round', type=int, default=0)
    parser.add_argument('--keyepoch', type=int, default=5, help='number of epoch to update key')
    """
    Used for model 
    """
    parser.add_argument('--model_type', type=str, default='ViT-B_16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained_dir', type=str, default='checkpoints/imagenet21k_ViT-B_16.npz', help='The pretrain model path')
    parser.add_argument('--n_prompt', type=int, default=4, help='The length of shared prompt')  
    parser.add_argument('--cls_num', type=int, default=10) 
    parser.add_argument('--share_blocks', nargs='+', type=int, default=[], help="shared transformer set 6 ")
    parser.add_argument('--share_blocks_g', nargs='+', type=int,  default=[], help="shared transformer set 6 ")
    
    parser.add_argument('--prompt_sample_type', type=str, default='max_pooling')
    parser.add_argument('--download', type=bool, default='True')
    parser.add_argument('--pin_memory', type=bool, default='True')
    parser.add_argument('--workers', type=int, default=0,help = 'number of threads')
    parser.add_argument('--client_num_per_dataset', type=int, default=4)
    parser.add_argument('--val_percent', type=float, default=0.15)
    parser.add_argument('--client_gam', type=int, default=15)

    parser.add_argument('--client_epochs', type=int, default=10)
    parser.add_argument('--server_epochs', type=int, default=5)
    parser.add_argument('--cross_depth', type=int, default=1)
    parser.add_argument('--num_cross_attention_heads', type=int, default=12)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    parser.add_argument('--ffn_dropout', type=float, default=0.2)
    parser.add_argument('--use_correlation_loss', type=bool, default=True)
    parser.add_argument('--use_dac_loss', type=bool, default=True)
    parser.add_argument('--use_knowledge', type=bool, default=True)
    parser.add_argument('--use_client_loss', type=bool, default=True)

    args = parser.parse_args()
                             
    return args

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)  
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass
def criterion_of_client(net,lambda_l2 = 0.01):
    prompt_common = []
    prompt_uncommon = []

    for i in net.share_blocks:
        prompt_common.append(net.prompt_common[net.share_blocks.index(i)])
    for i in net.share_blocks_g:
        prompt_uncommon.append(net.prompt_uncommon[net.share_blocks_g.index(i)])
    criterion = 0
    for i in range(len(prompt_common)):
        for j in range(len(prompt_uncommon)):
            l2_loss = torch.nn.MSELoss()(prompt_common[i], prompt_uncommon[j])
            l2_reg = lambda_l2 * (torch.norm(prompt_common[i]) ** 2 + torch.norm(prompt_uncommon[j]) ** 2)
            criterion += l2_loss + l2_reg
    criterion = criterion / (len(prompt_common) * len(prompt_uncommon))
    return criterion

def test_loss(net,args,param_dict):
    train_dataloader = param_dict['train_dataloader']
    lr = args.client_lr
    criterion = nn.CrossEntropyLoss().to(args.device)
    with torch.no_grad():
        x, target, _ = next(iter(train_dataloader))
        x, target = x.to(args.device), target.to(args.device)
        target = target.long()
        output = net(x)   
        out = output['logits']
        loss1 = criterion(out,target)
        if args.use_client_loss:
            loss2 = criterion_of_client(net)  
        else:
            loss2 = 0.0
        loss_test = loss1 + args.client_gam*loss2
    print('Training loss of client is {}'.format(loss_test))
    torch.cuda.empty_cache()
    return lr

def train_of_client(net,args,param_dict,lr_mine):
    train_dataloader = param_dict['train_dataloader']
    lr = lr_mine
    net.train()
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD([p for k,p in net.named_parameters() if p.requires_grad  and  ('head' in k or 'prompt' in  k )], lr=lr,momentum=args.rho,weight_decay=args.reg)
    cnt = 0
    
    for epoch in range(args.client_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target,_) in enumerate(train_dataloader):
            x, target = x.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            target = target.long()
            output = net(x)   
            out = output['logits']
            loss1 = criterion(out,target)
            if args.use_client_loss:
                loss2 = criterion_of_client(net)  
            else:
                loss2 = 0.0
            loss = loss1 + args.client_gam*loss2 
            epoch_loss_collector.append(loss.item())  
            loss.backward()
            optimizer.step()
            cnt += 1
            
            if batch_idx % 40 == 0:
                print('Training loss of client is {}'.format(sum(epoch_loss_collector) / len(epoch_loss_collector)))
                epoch_loss_collector = []
            torch.cuda.empty_cache()
    
    return net

def criterion_dac(out, selected, tau = 5):
    ou = out.reshape(out.shape[0], -1)
    o1,o2 = ou.shape
    scale = o2**-0.5
    distance_matrix = torch.cdist(ou, ou, p=2)**2 
    exp_distance_matrix = torch.exp(distance_matrix * scale / tau)
    
    dac_loss = 0.0
    total = 0
    for index in selected:
        numrator = 0
        denumrator = 0
        total = 0
        for i in selected:
            if i == index:
                continue
            numrator += exp_distance_matrix[index,i]
        for j in range(o1):
            if j == index:
                continue
            total += exp_distance_matrix[index,j]
        denumrator = total - numrator
        dac_loss += torch.log(numrator/denumrator)
    dac_loss = -dac_loss/len(selected)
    return dac_loss

def criterion_correlation(model,args):
    criterioning = nn.MSELoss()
    n_idt = int(len(args.share_blocks_g)*args.n_parties)
    idt = torch.eye(n=n_idt)
    gt = torch.zeros_like(idt)
    corr_p = model.knowledge_bank
    corr_p = corr_p.to(args.device)
    idt = idt.to(args.device)
    gt = gt.to(args.device)
    b1,b2,b3 = corr_p.shape
    corr_p = corr_p.view(b1,b2*b3)
    corr_transpose = torch.transpose(corr_p,0,1)
    BB_T = torch.matmul(corr_p, corr_transpose)
    BB_T_diag = BB_T * idt
    corr_loss = criterioning(BB_T-BB_T_diag,gt)
    return corr_loss

def aggregation_of_mine(net,prompt_commoning,prompt_uncommoning,selected,fed_avg_freqs,args,optimizer_corr=None,optimizer_dac=None,optimizer=None):
    global_prompt_common = torch.zeros_like(prompt_commoning[0])
    global_prompt_uncommon = torch.zeros_like(prompt_uncommoning[0])

    if args.use_knowledge:
        print("knowledge_bank.shape",net.cross_blocks.knowledge_bank.shape)
        s1,s2,s3,s4 = prompt_uncommoning.shape
        prompt_of_selected = prompt_uncommoning.clone().view(s1*s2,s3,s4)
        prompt_of_selected = prompt_of_selected.to(args.device)
        net.cross_blocks.to(args.device)    
        for epoch in range(args.server_epochs):
            new_knowledge = net.cross_blocks(prompt_of_selected)

            if args.use_correlation_loss and args.use_dac_loss:
                optimizer_corr.zero_grad()
                optimizer_dac.zero_grad()
                corr_loss = criterion_correlation(net.cross_blocks,args)
                dac_loss = criterion_dac(new_knowledge,selected)

                corr_loss.backward()
                dac_loss.backward()
                optimizer_corr.step()
                optimizer_dac.step()

            elif args.use_correlation_loss and not args.use_dac_loss:
                optimizer_corr.zero_grad()
                corr_loss = criterion_correlation(net.cross_blocks,args)
                corr_loss.backward()
                optimizer_corr.step()
            elif not args.use_correlation_loss and args.use_dac_loss:
                optimizer_dac.zero_grad()
                dac_loss = criterion_dac(new_knowledge,selected)
                dac_loss.backward()
                optimizer_dac.step()
            torch.cuda.empty_cache()
        net.cross_blocks.new_prompt.data = new_knowledge

        sh1,sh2,sh3 = new_knowledge.shape
        new_knowledge_reshaped = net.cross_blocks.new_prompt.data.view(args.n_parties, len(args.share_blocks_g), sh2, sh3)
        for i in range(args.n_parties):
            prompt_uncommoning[i] = new_knowledge_reshaped[i]
        for i in range(len(selected)):
            global_prompt_common += prompt_commoning[selected[i]] * fed_avg_freqs[i]
        for i in range(len(selected)):
            prompt_commoning[selected[i]] = global_prompt_common
         
    else:
        for i in range(len(selected)):
            global_prompt_common += prompt_commoning[selected[i]] * fed_avg_freqs[i]
            global_prompt_uncommon += prompt_uncommoning[selected[i]] * fed_avg_freqs[i]
        for i in range(len(selected)):
            prompt_commoning[selected[i]] = global_prompt_common
            prompt_uncommoning[selected[i]] = global_prompt_uncommon
    return net

if __name__ == '__main__':
    args = get_args()

    if args.dataset == 'CIFAR-100':
        cls_coarse = \
                np.array([
                    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
                ])
    save_path = args.model_type+"-"+str(args.n_parties)+"-"+args.dataset+"-"+str(args.cls_num)
    root_path = args.logdir
    save_path = Path(os.path.join(root_path,save_path))
    save_path.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_path,'commline.log'), sys.stdout)

    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print(save_path) 
    
    with open(os.path.join(save_path,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset == 'cifar100':
        X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
                    args.dataset, args.datadir, args.partition, args.n_parties, logdir=args.logdir,args=args)
    arr = np.arange(args.n_parties)

    if args.dataset == 'office':
        data_loader_dict,net_dataidx_map_train = prepare_data(args)
        num_classes = 10
    elif args.dataset == 'domainnet':
        data_loader_dict,net_dataidx_map_train = prepare_data_domain(args)
        num_classes = 10
    else: 
        data_loader_dict = {}
        for net_id in arr:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            data_loader_dict[net_id] = {}
            train_dl_local, test_dl_local, _, _ ,_,_ = get_divided_dataloader(args, dataidxs_train, dataidxs_test,traindata_cls_counts=traindata_cls_counts[net_id])
            num_classes = 100
            data_loader_dict[net_id]['train_dl_local'] = train_dl_local
            data_loader_dict[net_id]['test_dl_local'] = test_dl_local
                
    device = args.device
    config = CONFIGS[args.model_type]
    net = VisionTransformer_prompt(config,args.img_size, num_classes=num_classes,vis = True,args= args)
    net.load_from(np.load(args.pretrained_dir))
    net.freeze()
    net.to(device)

    prompt_commoning = []
    prompt_uncommoning = []
    keys_dict = {}
    dict_loss = {}
    client_dicts= torch.arange(args.n_parties)
    results_dict = defaultdict(list)

    for net_id in arr:
        prompt_commoning.append([torch.zeros(args.n_prompt, config.hidden_size) for _ in range(len(args.share_blocks))])
        prompt_uncommoning.append([torch.zeros(args.n_prompt, config.hidden_size) for _ in range(len(args.share_blocks_g))])
    prompt_commoning = torch.stack([torch.stack(client_prompts) for client_prompts in prompt_commoning])
    if args.share_blocks_g != []:
        prompt_uncommoning = torch.stack([torch.stack(client_prompts) for client_prompts in prompt_uncommoning])
    
    load_or_not = []
    for i in range(args.n_parties):
        load_or_not.append(1)
    
    for ix in range(len(arr)):
        net.load_prompt(prompt_commoning[ix],prompt_uncommoning[ix])
        train_dl_local = data_loader_dict[ix]['train_dl_local']
        x, target, _ = next(iter(train_dl_local))
        prompt_generator(net,x,args)
        prompt_commoning[ix], prompt_uncommoning[ix] = net.extract_prompt()
    if args.share_blocks_g != []:
        ha1,ha2,ha3,ha4 = prompt_uncommoning.shape
        prompt_uncommoning = prompt_uncommoning.to(device)
    if args.use_knowledge:
        net.cross_blocks.knowledge_bank.data = prompt_uncommoning.clone().view(ha1*ha2,ha3,ha4)
    
    optimizer_corr,optimizer_dac,optimizer_server = None,None,None
    if args.use_correlation_loss:
        optimizer_corr = optim.SGD([p for k,p in net.cross_blocks.named_parameters() if p.requires_grad  and ('knowledge_bank' in k ) ], lr=args.corr_lr, momentum=args.rho, weight_decay=args.reg)
        scheduler = lr_scheduler.StepLR(optimizer_corr, step_size=10, gamma=0.5)
    if args.use_dac_loss:
        optimizer_dac = optim.SGD([p for k,p in net.cross_blocks.named_parameters() if p.requires_grad  and  ('knowledge_bank' not in k ) ], lr=args.dac_lr, momentum=args.rho, weight_decay=args.reg)
    
    warmup_epochs = 20
    warmup_lr_init = 0.00001
    warmup_lr_end = 0.0001

    for round in range(warmup_epochs):
        print('########### Now is the warm round {} ######'.format(round))
        warmup_lr = warmup_lr_init + (warmup_lr_end - warmup_lr_init) * round / warmup_epochs  
        print("warmup_lr",warmup_lr)  
        if args.use_correlation_loss:
            for param_group in optimizer_corr.param_groups:
                param_group['lr'] = warmup_lr

        np.random.shuffle(arr)
        print("arr",arr)
        selected = arr[:int(args.n_parties * args.sample)]
        print("selected",selected)
        for ix in range(len(selected)):
            param_dict = {}
            idx = selected[ix]
            print('Now is the client {}'.format(idx))
            print("load_or_not[idx]",load_or_not[idx])
            train_dl_local = data_loader_dict[idx]['train_dl_local']
            test_dl_local = data_loader_dict[idx]['test_dl_local'] 
            param_dict['train_dataloader'] = train_dl_local
            param_dict['dict_loss'] = dict_loss
            net.load_prompt(prompt_commoning[idx],prompt_uncommoning[idx])
            lr_mine = test_loss(net,args,param_dict)
            net = train_of_client(net,args,param_dict,lr_mine)  
            prompt_commoning[idx], prompt_uncommoning[idx] = net.extract_prompt()
        
        total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]
        
        net = aggregation_of_mine(net,prompt_commoning,prompt_uncommoning,selected,fed_avg_freqs,args, optimizer_corr, optimizer_dac, optimizer_server)
        if (round+1)>=args.test_round:
            test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_our(net,data_loader_dict,args)
            print('>> Mean Local Test accuracy: %f' % local_mean_acc)
            print('>> Min Local Test accuracy: %f' % local_min_acc)
            print('>> Global Model Test accuracy: %f' % test_avg_acc)
            print('>> Test avg loss: %f' %test_avg_loss)

            results_dict['test_avg_loss'].append(test_avg_loss)
            results_dict['test_avg_acc'].append(test_avg_acc)
            results_dict['local_mean_acc'].append(local_mean_acc)
            results_dict['local_min_acc'].append(local_min_acc)

            outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
            torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)
        if args.use_correlation_loss:
            scheduler.step()
    for round in range(args.comm_round):
        print('########### Now is the round {} ######'.format(round))
        np.random.shuffle(arr)
        print("arr",arr)
        selected = arr[:int(args.n_parties * args.sample)]
        print("selected",selected)
        for ix in range(len(selected)):
            param_dict = {}
            idx = selected[ix]
            print('Now is the client {}'.format(idx))
            print("load_or_not[idx]",load_or_not[idx])
            train_dl_local = data_loader_dict[idx]['train_dl_local']
            test_dl_local = data_loader_dict[idx]['test_dl_local'] 
            param_dict['train_dataloader'] = train_dl_local
            param_dict['dict_loss'] = dict_loss

            net.load_prompt(prompt_commoning[idx],prompt_uncommoning[idx])
            lr_mine = test_loss(net,args,param_dict)
            net = train_of_client(net,args,param_dict,lr_mine)
            prompt_commoning[idx], prompt_uncommoning[idx] = net.extract_prompt()
        total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]
        net = aggregation_of_mine(net,prompt_commoning,prompt_uncommoning,selected,fed_avg_freqs,args, optimizer_corr, optimizer_dac, optimizer_server)
        if (round+1)>=args.test_round:
            test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy_our(net,data_loader_dict,args)
            print('>> Mean Local Test accuracy: %f' % local_mean_acc)
            print('>> Min Local Test accuracy: %f' % local_min_acc)
            print('>> Global Model Test accuracy: %f' % test_avg_acc)
            print('>> Test avg loss: %f' %test_avg_loss)
            results_dict['test_avg_loss'].append(test_avg_loss)
            results_dict['test_avg_acc'].append(test_avg_acc)
            results_dict['local_mean_acc'].append(local_mean_acc)
            results_dict['local_min_acc'].append(local_min_acc)
            outfile_vit = os.path.join(save_path, 'Vit_1500_round{}.tar'.format(round))
            torch.save({'epoch':args.comm_round+1, 'state':net.state_dict()}, outfile_vit)
        print("load_or_not",load_or_not)
        if args.use_correlation_loss:
            scheduler.step()