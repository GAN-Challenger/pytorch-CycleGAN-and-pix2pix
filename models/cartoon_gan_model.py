import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
#import torchvision.models as models

class CartoonGANModel(BaseModel):
    def name(self):
        return "CartoonGANModel"
    
    def initialize(self,opt):
        print("------cartoon gan initialize------")
        self.batchsize = opt.batchSize
        BaseModel.initialize(self,opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
	
        self.netD = networks.define_D(opt.output_nc, opt.ndf,opt.which_model_netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D', which_epoch)
        #print self.isTrain
        if self.isTrain:
            #self.real = ImagePool(opt.pool_size)
            #self.fake = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionContent = networks.ContentLoss(opt.content_loss,opt.lambda_content,opt.gpu_ids[0])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG.parameters()),lr=0.0005, momentum=0.7)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG, opt.verbose)
        if self.isTrain:
            networks.print_network(self.netD, opt.verbose)
        print('-----------------------------------------------')				

    def set_input(self, input):
        #AtoB = self.opt.which_direction == 'AtoB'
        #print("call set_input!")
        input_C = input["C"]
        input_S = input["S"]
        input_E = input["E"]
        if len(self.gpu_ids) > 0:
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            input_S = input_S.cuda(self.gpu_ids[0], async=True)
            input_E = input_E.cuda(self.gpu_ids[0], async=True)
        self.input_C = input_C
        self.input_S = input_S
        self.input_E = input_E		
        #self.image_paths = input['A_paths' if AtoB else 'B_paths')

    def forward(self):
        #print("call forward!")
        self.content = Variable(self.input_C)		
        self.style = Variable(self.input_S)
        self.edge = Variable(self.input_E)
    def vggInput(self,input):
        l = []
        for i in range(input.size()[0]):
            r = torch.add(input[i][0],-0.485)
            r = torch.div(r,0.229)
            r = r.view(1,1,r.size()[0],r.size()[1])

            g = torch.add(input[i][1],-0.456)
            g = torch.div(g,0.224)
            g = g.view(1,1,g.size()[0],g.size()[1])

            b = torch.add(input[i][2],-0.406)
            b = torch.div(b,0.225)
            b = b.view(1,1,b.size()[0],b.size()[1])

            rgb = torch.cat((r,g,b),1)
            l.append(rgb)

        return torch.cat(l,0)

    def backward_G_content(self):
        #print self.content
        self.gfake = self.netG(self.content)
        gfake = self.vggInput(self.gfake)
        content = self.vggInput(self.content)

        #print gfake.size()
        #print content.size()
        #print("call backward_g_content...")
        self.loss_C = self.criterionContent(gfake,content)
        self.loss_C.backward()
        self.loss_G = 0
        self.loss_D = 0

    def backward_G(self):
        # GAN loss D(G(A))
        self.gfake = self.netG(self.content)
        gfake = self.vggInput(self.gfake)
        content = self.vggInput(self.content)

        self.loss_C = self.criterionContent(gfake,content)
        self.loss_G = self.criterionGAN(self.netD(self.gfake), True)
        self.loss_C.backward()
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake, gfake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # gFake
        pred_gfake = netD(gfake.detach())
        loss_D_gfake = self.criterionGAN(pred_gfake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake + loss_D_gfake)/3.0
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #fake = self.fake_B_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.style, self.edge,self.gfake)

    def pre_optimize_parameters(self):
        self.forward()
        #initialize generator net
        self.optimizer_G.zero_grad()
        self.backward_G_content()
        self.optimizer_G.step()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D', self.loss_D), ('G', self.loss_G), ('C', self.loss_C)])
        return ret_errors

    def save(self,label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
