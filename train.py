import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import datetime
import numpy as np
import os
import pickle
import shutil
import tarfile
import urllib.request

from model import ReconstractionTICA
from visualize import visualize, visualize_kernel

# Define constants
λ = 1.0     # Reconstruction coefficient
N = 100     # Minibatch size
SNAPSHOT_INTERVAL = 10

def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/cifar-10-gray.npy'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/cifar-10-python.tar.gz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/cifar-10-python.tar.gz', 'r') as stream:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(stream, "dataset/")
        train = []
        for path in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
            path = 'dataset/cifar-10-batches-py/' + path
            with open(path, 'rb') as stream:
                _ = pickle.load(stream, encoding='bytes')
                _ = np.frombuffer(_[b'data'], dtype=np.uint8).reshape((-1, 3, 32, 32))
            for rgb in _:
                r = rgb[0, :, :].astype('f') / 255.
                g = rgb[1, :, :].astype('f') / 255.
                b = rgb[2, :, :].astype('f') / 255.
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                train.append(gray)
        train = np.array(train, dtype='f')
        np.save('dataset/cifar-10-gray', train)
    os.remove('dataset/cifar-10-python.tar.gz') if os.path.exists('dataset/cifar-10-python.tar.gz') else None
    shutil.rmtree('dataset/cifar-10-batches-py', ignore_errors=True)

    # (Download CIFAR100 dataset)
    if not os.path.exists('dataset/cifar-100-gray.npy'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/cifar-100-python.tar.gz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/cifar-100-python.tar.gz', 'r') as stream:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(stream, "dataset/")
        train = []
        for path in ['train', 'test']:
            path = 'dataset/cifar-100-python/' + path
            with open(path, 'rb') as stream:
                _ = pickle.load(stream, encoding='bytes')
                _ = np.frombuffer(_[b'data'], dtype=np.uint8).reshape((-1, 3, 32, 32))
            for rgb in _:
                r = rgb[0, :, :].astype('f') / 255.
                g = rgb[1, :, :].astype('f') / 255.
                b = rgb[2, :, :].astype('f') / 255.
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                train.append(gray)
        train = np.array(train, dtype='f')
        np.save('dataset/cifar-100-gray', train)
    os.remove('dataset/cifar-100-python.tar.gz') if os.path.exists('dataset/cifar-100-python.tar.gz') else None
    shutil.rmtree('dataset/cifar-100-python', ignore_errors=True)
    
    # Create samples.
    train = np.load('dataset/cifar-10-gray.npy').reshape((-1, 1, 32, 32))   # cifar-10 or cifar-100
    train = np.random.permutation(train)
    validation = train[0:100]

    # Create the model
    nn = ReconstractionTICA()

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    nn.to_gpu()

    # Setup optimizers
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(nn)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # (Validate input images)
    visualize(validation, 'validation.png', (32, 32))

    # Training
    for epoch in range(200):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)
            y, z = nn(chainer.cuda.to_gpu(validation))
            visualize(chainer.cuda.to_cpu(z.data), 'z.png', (32, 32))
            visualize(chainer.cuda.to_cpu(y.data), 'y.png', (18, 18))
            visualize_kernel(chainer.cuda.to_cpu(nn.f.W.data), 'W.png')
            os.chdir('..')

        # (Random shuffle samples)
        train = np.random.permutation(train)

        total_loss_reg = 0.0
        total_loss_rec = 0.0

        for n in range(0, len(train), N):
            x = chainer.cuda.to_gpu(train[n:n + N].reshape((N, 1 * 32 * 32)))
            y, z = nn(x)
            # loss_reg = F.sum(y)
            # loss_rec = F.sum((x - z) ** 2)
            loss_reg = F.mean(y)
            loss_rec = F.sum((x - z) ** 2) / np.prod(x.shape)
            loss = loss_reg + λ * loss_rec
            nn.cleargrads()
            loss.backward()
            optimizer.update()

            total_loss_reg += loss_reg.data
            total_loss_rec += loss_rec.data

        # (View loss)
        total_loss_reg /= len(train) / N
        total_loss_rec /= len(train) / N
        print(epoch, total_loss_reg, total_loss_rec)


if __name__ == '__main__':
    main()
