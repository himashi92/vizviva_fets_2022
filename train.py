import argparse
import os
import random
import pathlib
import time
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from fvcore.nn import FlopCountAnalysis
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.brats import get_datasets
from learning_rate.poly_lr import poly_lr
from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from loss.vat import vat_loss
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, generate_segmentations_monai
from crswin2.vision_transformer import VTUNet as ViT_seg

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='CR-Swin2-VT FETS 2022 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/cr_swin2_vt.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=True, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.deterministic:
        cudnn.benchmark = True
        cudnn.enabled = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.exp_name = "logs"
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./saved_model")

    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args.cfg)
    model_1 = ViT_seg(config, num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    model_1.load_from(config)

    if args.resume:
        args.checkpoint = args.checkpoint_folder / "cr_swin_2.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1, device)

    print(f"total number of trainable parameters {count_parameters(model_1)}")

    model_1 = model_1.cuda()
    input = torch.rand(1, 4, 128, 128, 128).cuda()
    flops = FlopCountAnalysis(model_1, input)
    print(f"total number of flops {flops.total()}")

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss_Val().cuda()
    metric = criterian_val.metric
    print(metric)
    params = model_1.parameters()

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    full_train_dataset, val_dataset, test_dataset = get_datasets(args.seed, fold_number=args.fold)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    bench_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    print("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    best_1 = 0.0
    patients_perf = []

    print("start training now!")

    for epoch in range(args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")

            end = time.perf_counter()
            metrics = []

            optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.epochs, args.lr, 0.9)

            for i, batch in enumerate(zip(train_loader)):
                torch.cuda.empty_cache()
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()

                segs_S1 = model_1(inputs_S1)

                loss_dice = criterion(segs_S1, labels_S1)
                loss_vat = vat_loss(model_1, inputs_S1, labels_S1, eps=2.5).cuda()

                loss_ = loss_dice + 0.2 * loss_vat

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                # compute gradient and do SGD step
                loss_.backward()
                optimizer.step()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                          save_folder=args.save_folder_1,
                                                          patients_perf=patients_perf)

                t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

                if validation_dice > best_1:
                    print(f"Saving the model with DSC {validation_dice}")
                    best_1 = validation_dice
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            generate_segmentations_monai(bench_loader, model_1, t_writer_1, args)
            break

    generate_segmentations_monai(bench_loader, model_1, t_writer_1, args)


def step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["patient_id"]

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )
            val_outputs = inference(val_inputs, model)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]

            segs = val_outputs
            targets = val_labels
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return dice_values


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
