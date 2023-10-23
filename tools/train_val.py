import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
import torch


parser = argparse.ArgumentParser(description='Monocular 3D Object Detection')
# config choice
# parser.add_argument('--config', dest='config',default = '/home/XXXX/code/MonoSTL/MonoSTL_MonoDLE/experiments/example/kitti_example_centernet_depth.yaml', help='settings of detection in yaml format')
# parser.add_argument('--config', dest='config',default = '/home/XXXX/code/MonoSTL/MonoSTL_MonoDLE/experiments/example/kitti_example_centernet.yaml', help='settings of detection in yaml format')
parser.add_argument('--config', dest='config',default = '/home/XXXX/code/MonoSTL/MonoSTL_MonoDLE/experiments/example/kitti_example_distill.yaml', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')

args = parser.parse_args()

def copy_import_files(cfg):
    
    ckpt_name = os.path.join('log',cfg['trainer']['model_save_path']+'_save_files')
    
    if os.path.exists(ckpt_name) :
        shutil.rmtree(ckpt_name)
    
    os.makedirs(ckpt_name, exist_ok=True)
    
    shutil.copytree('experiments', ckpt_name + '/experiments')
    shutil.copytree('lib', ckpt_name + '/lib')
    shutil.copytree('tools', ckpt_name + '/tools')

def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    log_path = '/home/XXXX/code/MonoSTL/MonoSTL_MonoDLE/log/'
    if os.path.exists(log_path):
        pass
    else:
        os.mkdir(log_path)
    log_file = str(cfg['trainer']['model_save_path'])+'_%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S%d') +'.log'
    logger = create_logger(log_path, log_file)


    # build dataloader
    train_loader, test_loader,trainval_loader,test_submit_loader  = build_dataloader(cfg['dataset'])

    # test
    # args.evaluate_only = True
    if args.evaluate_only:
        model = build_model(cfg['model'], 'testing')

        logger.info('###################  Evaluation Only  ##################')
        if cfg['trainer']['spilt'] == 'trainval':
            tester = Tester(cfg=cfg['tester'],
                            model=model,
                            dataloader=test_submit_loader,
                            logger=logger)
        else:
            tester = Tester(cfg=cfg['tester'],
                            model=model,
                            dataloader=test_loader,
                            logger=logger)
        tester.test()
        return

    # build model&&build optimizer
    if cfg['model']['type']=='centernet3d' or cfg['model']['type']=='centernet3d_depth' or cfg['model']['type']=='distill':
        model = build_model(cfg['model'],'training')
        optimizer = build_optimizer(cfg['optimizer'], model)
        lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
        logger.info(lr_scheduler)
    
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])

    copy_import_files(cfg)

    logger.info('random_seed:'+str(cfg['random_seed']))
    logger.info(torch.__version__)
    logger.info('gpu:'+os.environ['CUDA_VISIBLE_DEVICES'])
    logger.info('pid:'+ str(os.getpid()))
    logger.info(cfg['trainer']['model_save_path'])
    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    
    if cfg['trainer']['spilt'] == 'trainval':
        trainer = Trainer(cfg=cfg['trainer'],   
                      model=model,
                      optimizer=optimizer,
                      train_loader=trainval_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      model_type=cfg['model']['type'],
                      root_path=ROOT_DIR)
        trainer.train()
    else:
        trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      model_type=cfg['model']['type'],
                      root_path=ROOT_DIR)
        trainer.train()

        logger.info('###################  Evaluation  ##################' )
        test_model_list = build_model(cfg['model'],'testing')
        tester = Tester(cfg=cfg['tester'],
                    model=test_model_list,
                    dataloader=test_loader,
                    logger=logger)
        tester.test()


if __name__ == '__main__':
    main()