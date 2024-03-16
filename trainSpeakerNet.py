#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from log.controller import LogModuleController

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--eval_frames_for_short',    type=str,   default="",    help='Input length to the network for testing short utternaces')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=120,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="cosine_warmup", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=3e-6,  help='Learning rate')
parser.add_argument('--T_0',             type=float, default=60,  help='Learning rate')
parser.add_argument('--T_mult',             type=float, default=1,  help='Learning rate')
parser.add_argument('--eta_max',             type=float, default=5e-4,  help='Learning rate')
parser.add_argument('--gamma',             type=float, default=0.5,  help='Learning rate')

parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.3,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=7205,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exp/model", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_path',     type=str,   default="/workspace/MR-RawNet/DB", help='Absolute path to the train set')
parser.add_argument('--train_list',     type=str,   default="/workspace/MR-RawNet/DB/train_list_v12.txt",  help='Train list')
parser.add_argument('--musan_path',     type=str,   default="/workspace/MR-RawNet/DB/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/workspace/MR-RawNet/DB/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
parser.add_argument('--test_voxceleb1_path',      type=str,   default="/workspace/MR-RawNet/DB/VoxCeleb1/test", help='Absolute path to the test set')
parser.add_argument('--test_voxceleb1_list',      type=str,   default="/workspace/MR-RawNet/DB/VoxCeleb1/trials/trials.txt",   help='Evaluation list')
parser.add_argument('--test_voices_dev_path',     type=str,   default="", help='Absolute path to the VOiCES test set')
parser.add_argument('--test_voices_dev_list',     type=str,   default="",   help='Absolute path to the VOiCES Evaluation list')
#parser.add_argument('--test_voices_dev_path',     type=str,   default="/workspace/MR-RawNet/DB/VOiCES/Development_Data/Speaker_Recognition/sid_dev")
#parser.add_argument('--test_voices_dev_list',     type=str,   default="/workspace/MR-RawNet/DB/VOiCES/Development_Data/Speaker_Recognition/sid_dev_lists_and_keys/dev-trial-keys.txt")


## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8855", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

### For logging
project = 'MR-RawNet'
tag = '01' 
name = 'MR-RawNet'
wandb_group = ''
wandb_entity = ''
wandb_api_key = ''
neptune_token = ''
neptune_user = ''

parser.add_argument('--name', default=name, type = str)
parser.add_argument('--tags',default=tag, type = str)
parser.add_argument('--project',default=project, type = str)
parser.add_argument('--path_logging', default='exps/', type = str)
parser.add_argument('--path_scripts', default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument('--wandb_group', default=wandb_group, type = str)
parser.add_argument('--wandb_entity', default=wandb_entity, type = str)
parser.add_argument('--wandb_api_key', default=wandb_api_key, type = str)
parser.add_argument('--neptune_user', default=neptune_user, type = str)
parser.add_argument('--neptune_token', default=neptune_token, type = str)

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    args.number_iteration = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    

    # For logging
    if args.gpu == 0:
        logger = LogModuleController.Builder(args.name, args.project
            ).tags([args.tags]
            ).save_source_files(args.path_scripts
            ).use_local(args.path_logging
            #).use_wandb(args.wandb_group, args.wandb_entity, args.wandb_api_key
            #).use_neptune(args.neptune_user, args.neptune_token
            ).build()
        logger.log_parameter(vars(args))
    else:
        logger = None

    trainer     = ModelTrainer(s, logger, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()


    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ',pytorch_total_params)

        scorefile.write("   VoxCeleb1   \n")
        sc, lab, _, feats = trainer.evaluateFromList(test_path=args.test_voxceleb1_path, test_list=args.test_voxceleb1_list, **vars(args))
        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            
            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print("\nVEER {:2.4f}, MinDCF {:2.5f}".format(result[1], mindcf))
            scorefile.write("VEER {:2.4f}, MinDCF {:2.5f}\n".format(result[1], mindcf))
            

        if args.eval_frames_for_short:
            for eval_frames in args.eval_frames_for_short.split(','):
                sc, lab, _, _ = trainer.evaluateFromList(test_path=args.test_voxceleb1_path, test_list=args.test_voxceleb1_list, 
                    shortmode=int(eval_frames), ref_feats=feats, **vars(args))

                if args.gpu == 0:
            
                    result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                    print("\nVEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}".format(eval_frames, result[1], eval_frames, mindcf))
                    scorefile.write("VEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}\n".format(eval_frames, result[1], eval_frames, mindcf))
                    scorefile.flush()


        if args.test_voices_dev_path and args.test_voices_dev_list:
            scorefile.write("   VOiCES Dev   \n")
            sc, lab, _, feats = trainer.evaluateFromList(test_path=args.test_voices_dev_path, test_list=args.test_voices_dev_list, **vars(args))

            if args.gpu == 0:

                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                
                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                print("\nVEER {:2.4f}, MinDCF {:2.5f}".format(result[1], mindcf))
                scorefile.write("VEER {:2.4f}, MinDCF {:2.5f}\n".format(result[1], mindcf))

            if args.eval_frames_for_short:
                for eval_frames in args.eval_frames_for_short.split(','):
                    sc, lab, _, _ = trainer.evaluateFromList(test_path=args.test_voices_dev_path, test_list=args.test_voices_dev_list,
                        shortmode=int(eval_frames), ref_feats=feats, **vars(args))

                    if args.gpu == 0:
                
                        result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                        print("\nVEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}".format(eval_frames, result[1], eval_frames, mindcf))
                        scorefile.write("VEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}\n".format(eval_frames, result[1], eval_frames, mindcf))
                        scorefile.flush()


        if args.gpu == 0:
            scorefile.close()
            logger.finish()

        return


    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))


        if it % args.test_interval == 0:
            
            sc, lab, _, feats = trainer.evaluateFromList(test_path=args.test_voxceleb1_path, test_list=args.test_voxceleb1_list, **vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("   VoxCeleb1   \n")
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                logger.log_metric("EER", result[1], epoch_step = it)
                logger.log_metric("EER_BEST", min(eers), epoch_step = it)
                logger.log_metric("MinDCF", mindcf, epoch_step = it)
                logger.log_metric("lr", max(clr), epoch_step = it)
                
                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

            if args.eval_frames_for_short:
                for eval_frames in args.eval_frames_for_short.split(','):
                    sc, lab, _, _ = trainer.evaluateFromList(test_path=args.test_voxceleb1_path, test_list=args.test_voxceleb1_list, 
                        shortmode=int(eval_frames), ref_feats=feats, **vars(args))

                    if args.gpu == 0:
                
                        result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                        scorefile.write("Epoch {:d}, VEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}\n".format(it, eval_frames, result[1], eval_frames, mindcf))
                        logger.log_metric(f"EER_{eval_frames}", result[1], epoch_step = it)

                        scorefile.flush()
            
            ### VOiCES Dev   
            if args.test_voices_dev_path and args.test_voices_dev_list:     
                sc, lab, _, feats = trainer.evaluateFromList(test_path=args.test_voices_dev_path, test_list=args.test_voices_dev_list, **vars(args))

                if args.gpu == 0:
                    
                    result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)


                    print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                    scorefile.write("   VOiCES Dev   \n")
                    scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                    logger.log_metric("VOiCES_Dev_EER", result[1], epoch_step = it)

                    scorefile.flush()
            
                
                if args.eval_frames_for_short:
                    for eval_frames in args.eval_frames_for_short.split(','):
                        sc, lab, _, _ = trainer.evaluateFromList(test_path=args.test_voices_dev_path, test_list=args.test_voices_dev_list, 
                            shortmode=int(eval_frames), ref_feats=feats, **vars(args))

                        if args.gpu == 0:
                    
                            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                            scorefile.write("Epoch {:d}, VEER_{:s} {:2.4f}, MinDCF_{:s} {:2.5f}\n".format(it, eval_frames, result[1], eval_frames, mindcf))
                            logger.log_metric(f"VOiCES_Dev_EER_{eval_frames}", result[1], epoch_step = it)

                            scorefile.flush()

            

    if args.gpu == 0:
        scorefile.close()
        logger.finish()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)
    print(f"args.distributed: {args.distributed}")

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()