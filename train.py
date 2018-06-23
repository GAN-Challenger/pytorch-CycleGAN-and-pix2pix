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
    #visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print "epoch ",epoch
        loss = np.zeros((8))
        for i, data in enumerate(dataset):
            #print "..."
            if i % 100 == 0:
                print time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print i,"/",dataset_size

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            errors = model.get_current_errors()
            #print "D_A",errors["D_A"].data.cpu().numpy()," G_A",errors["G_A"].data.cpu().numpy()," Cyc_A",errors["Cyc_A"].data.cpu().numpy()," idt_A",errors["idt_A"].data.cpu().numpy()," D_B",errors["D_B"].data.cpu().numpy()," G_B",errors["G_B"].data.cpu().numpy()," Cyc_B",errors["Cyc_B"].data.cpu().numpy()," idt_B",errors["idt_B"].data.cpu().numpy()

            loss[0] += errors["D_A"].data.cpu().numpy()[0]
            loss[1] += errors["G_A"].data.cpu().numpy()[0]
            loss[2] += errors["Cyc_A"].data.cpu().numpy()[0]
            loss[3] += errors["idt_A"].data.cpu().numpy()[0]
	    
            loss[4] += errors["D_B"].data.cpu().numpy()[0]
            loss[5] += errors["G_B"].data.cpu().numpy()[0]
            loss[6] += errors["Cyc_B"].data.cpu().numpy()[0]
            loss[7] += errors["idt_B"].data.cpu().numpy()[0]

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                #visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                #if opt.display_id > 0:
                   # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        print loss
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
