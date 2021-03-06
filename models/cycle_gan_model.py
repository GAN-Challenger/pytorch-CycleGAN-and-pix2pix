import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
#import numpy as np
#from PIL import Image

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A, opt.verbose)
        networks.print_network(self.netG_B, opt.verbose)
        if self.isTrain:
            networks.print_network(self.netD_A, opt.verbose)
            networks.print_network(self.netD_B, opt.verbose)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        #print real_A
        self.fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(self.fake_B)

        real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def inferenceA2B(self,input_A):
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
        
        self.real_A = Variable(input_A,volatile=True)
        #print self.real_A
        self.fake_B = self.netG_A(self.real_A)

    def inferenceB2A(self,input_B):
        if len(self.gpu_ids) > 0:
            input_B = input_B.cuda(self.gpu_ids[0])

        self.real_B = Variable(input_B,volatile=True)
        self.fake_A = self.netG_B(self.real_B)

    def to_image(self,tensor):
        return util.tensor2im(tensor)
    def save_image(self,path,tensor):
        util.save_image(self.to_image(tensor),path)

    def save_result(self,path,name):
        image = self.to_image(self.input_A)
        util.save_image(image,path + "/" + name + "_a.jpg")

        image = self.to_image(self.input_B)
        util.save_image(image,path + "/" + name + "_b.jpg")

        image = self.to_image(self.fake_B)
        util.save_image(image,path + "/" + name + "_gb.jpg")

        image = self.to_image(self.rec_A)
        util.save_image(image,path + "/" + name + "_ra.jpg")
        #cv2.imwrite(path + "/" + name + "_ra.jpg",image)

        image = self.to_image(self.fake_A)
        util.save_image(image,path + "/" + name + "_ga.jpg")
        #cv2.imwrite(path + "/" + name + "_ga.jpg",image)

        image = self.to_image(self.rec_B)
        util.save_image(image,path + "/" + name + "_rb.jpg")
        #cv2.imwrite(path + "/" + name + "_rb.jpg",image)
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.fake_B = self.netG_A(self.real_A)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.fake_A = self.netG_B(self.real_B)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss
        self.rec_A = self.netG_B(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B)])
        if self.opt.lambda_identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict([('real_A', self.real_A), ('fake_B', self.fake_B), ('rec_A', self.rec_A),
                                   ('real_B', self.real_B), ('fake_A', self.fake_A), ('rec_B', self.rec_B)])
        if self.opt.isTrain and self.opt.lambda_identity > 0.0:
            ret_visuals['idt_A'] = self.idt_A
            ret_visuals['idt_B'] = self.idt_B
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
