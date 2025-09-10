import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--lrbl', type=float, default=1.0, help='learning rate of balance')
parser.add_argument('--epochw', type=int, default=10, help='number of epochs to train weight')  # 10

parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--lambdap', type=float, default=1, help='weight decay for weight1 ')

# for pow
parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')
