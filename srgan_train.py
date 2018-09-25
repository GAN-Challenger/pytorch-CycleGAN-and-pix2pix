import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    #model.initialize(opt)
    #visualizer = Visualizer(opt)
    for epoch in range(0):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print "epoch ",epoch
        loss = np.zeros((3))
        count = 0;
        for i, data in enumerate(dataset):
            #print "..."

            #iter_start_time = time.time()
            #if total_steps % opt.print_freq == 0:
            #t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            #total_steps += opt.batchSize
            #epoch_iter += opt.batchSize
            model.set_input(data)
            model.pre_optimize_parameters()

            errors = model.get_current_errors()
            print errors["C"].data.cpu().numpy()[0]
            loss[0] += errors["C"].data.cpu().numpy()[0]
            count += 1
            #loss[1] += errors["G"].data.cpu().numpy()[0]
            #loss[2] += errors["D"].data.cpu().numpy()[0]
        print time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        print loss/count

        model.save('latest')
        model.save(epoch)

    #train cartoon_net
    for epoch in range(100):
        #epoch_start_time = time.time()
        #iter_data_time = time.time()
        print "epoch ",epoch
        loss = np.zeros((3))
        count = 0
        for i, data in enumerate(dataset):
            #print "..."
            #iter_start_time = time.time()
            #if total_steps % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            #total_steps += opt.batchSize
            #epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            errors = model.get_current_errors()
            loss[0] += errors["C"].data.cpu().numpy()[0]
            loss[1] += errors["G"].data.cpu().numpy()[0]
            loss[2] += errors["D"].data.cpu().numpy()[0]

            print  errors["C"].data.cpu().numpy()[0],\
                   errors["G"].data.cpu().numpy()[0],\
                   errors["D"].data.cpu().numpy()[0]
            count += 1

        print time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        print loss/count
        
        model.save('latest')
        model.save(epoch)

        model.update_learning_rate()
