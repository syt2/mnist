import os, sys, time, random,torch
import torch.backends.cudnn as cudnn
import argparse
from pytorch_model_summary.model_summary import model_summary
from utils import AverageMeter, \
    RecorderMeter, time_string, \
    convert_secs2time,print_log,\
    accuracy,adjust_learning_rate,\
    save_checkpoint
import net
from tqdm import tqdm
import dataset
import tensorboardX
from tensorboardX import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser=argparse.ArgumentParser(description='MNIST',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--train_dir', type=str, default='dataset', help='train dir.')
parser.add_argument('--test_dir', type=str, default='dataset', help='test dir.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[15,24],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--save_path', type=str, default='./save', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard', default='', type=str, metavar='PATH', help='path to tensorboard writer (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=3752, help='manual seed')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dict or not')

args=parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main(arch=None):

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, '{}.txt'.format('log')), 'w')

    if args.tensorboard is None:
        writer=SummaryWriter(args.save_path)
    else:
        writer=SummaryWriter(args.tensorboard)

    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("use cuda: {}".format(args.use_cuda), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Init data loader
    train_loader = dataset.mnistDataLoader(args.train_dir, True, args.train_batch_size, True, args.workers)
    test_loader = dataset.mnistDataLoader(args.test_dir, False, args.test_batch_size, False, args.workers)
    num_classes=10
    input_size=(1,28,28)
    net=arch(num_classes)
    print_log("=> network:\n {}".format(net),log)
    summary=model_summary(net,input_size)
    print_log(summary,log)

    writer.add_graph(net,torch.rand([1,1,28,28]))

    if args.ngpu>1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint=torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> not use any checkpoint for model", log)

    if args.evaluate:
        checkpoint=torch.load(args.save_path+'/model_best.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        time1 = time.time()
        validate(test_loader, net, criterion,log,writer,embedding=True)
        time2 = time.time()
        print('validate function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    start_time=time.time()
    epoch_time=AverageMeter()
    for epoch in range(args.start_epoch,args.epochs):
        current_learning_rate=adjust_learning_rate(args.learning_rate,optimizer,epoch,args.gammas,args.schedule)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}]'.format(recorder.max_accuracy(False)), log)
        train_acc, train_los = train(train_loader, net, criterion, optimizer,log)
        val_acc, val_los = validate(test_loader, net, criterion,log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            },is_best,args.save_path,'checkpoint.pth.tar')
            print('save ckpt done!')

        writer.add_scalar('Train/loss',train_los,epoch)
        writer.add_scalar('Train/acc',train_acc,epoch)
        writer.add_scalar('Test/acc',val_acc,epoch)
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    save_checkpoint({
        'state_dict': net.state_dict(),
        'recorder': recorder,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path, 'model.pth.tar')
    print('save model done!')


    checkpoint = torch.load(args.save_path + '/model_best.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    time1 = time.time()
    validate(test_loader, net, criterion, log, writer,embedding=True)
    time2 = time.time()
    print_log('validate function took %0.3f ms' % ((time2 - time1) * 1000.0),log)

    log.close()
    writer.close()




def train(train_loader,model,criterion,optimizer,log=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end_time=time.time()
    for i, (input, label) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end_time)
        if args.use_cuda:
            label = label.cuda()
            input = input.cuda()
        with torch.no_grad():
            input_var=torch.autograd.Variable(input)
            label_var=torch.autograd.Variable(label)
        output=model(input_var)
        loss=criterion(output,label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end_time)
        end_time=time.time()

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5),log)
    return top1.avg, losses.avg

def validate(val_loader,model,criterion,log=None,writer=None,embedding=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i,(input,label) in enumerate(tqdm(val_loader)):
        if args.use_cuda:
            label=label.cuda()
            input=input.cuda()
        with torch.no_grad():
            input_var=torch.autograd.Variable(input)
            label_var=torch.autograd.Variable(label)
        output=model(input_var)
        loss=criterion(output,label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        if embedding and writer is not None and i==0:
            out=torch.cat((output.cpu().data,torch.ones(len(output),1)),1)
            writer.add_embedding(out,metadata=label.data,label_img=input.data)

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5),log)
    return top1.avg, losses.avg


if __name__ == '__main__':
    args.save_path='./save'
    args.resume=args.save_path+'/checkpoint.pth.tar'
    args.tensorboard=args.save_path+'/tensorboard'
    # args.evaluate=True
    main(net.CNN)