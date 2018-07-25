import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2

#the input image : content_image target_image target_no_edge_image
class CartoonDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        #A is the content B is the style
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_S = os.path.join(opt.dataroot, opt.phase + 'S')

        self.C_paths = make_dataset(self.dir_C)
        self.S_paths = make_dataset(self.dir_S)

        self.C_paths = sorted(self.C_paths)
        self.S_paths = sorted(self.S_paths)
        self.C_size = len(self.C_paths)
        self.S_size = len(self.S_paths)
        self.transform = get_transform(opt)
        #content image		
        self.list_C = range(self.C_size)
        random.shuffle(self.list_C)
        self.index_C = 0
        #style image
        self.list_S = range(self.S_size)
        random.shuffle(self.list_S)
        self.index_S = 0

        #style image no edge
        self.list_E = range(self.S_size)
        random.shuffle(self.list_E)
        self.index_E = 0

        print("------------cartoon_gan---------------")
        print("cartoon gan dataset initialize finish!")
        print("------------cartoon_gan---------------")

    def __getitem__(self, index):
        if self.index_C >= self.C_size:
            random.shuffle(self.list_C)
            self.index_C = 0

        if self.index_S >= self.S_size:
            random.shuffle(self.list_S)
            self.index_S = 0

        if self.index_E >= self.S_size:
            random.shuffle(self.list_E)
            self.index_E = 0 			

        C_path = self.C_paths[self.list_C[self.index_C]]
        self.index_C += 1
		
        S_path = self.S_paths[self.list_S[self.index_S]]
        self.index_S += 1

        E_path = self.S_paths[self.list_E[self.index_E]]
        self.index_E += 1		
        #B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        C_img = Image.open(C_path).convert('RGB')
        S_img = Image.open(S_path).convert('RGB')

        cv2_im = cv2.imread(E_path)
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        cv2_im = cv2.blur(cv2_im, (5,5))
        E_img = Image.fromarray(cv2_im)

        C = self.transform(C_img)
        S = self.transform(S_img)
        E = self.transform(E_img)
        """
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        """
        return {'C': C, 'S': S, "E": E,'C_paths': C_path,'S_paths': S_path,'E_paths': E_path}

    def __len__(self):
        return max(self.C_size, self.S_size)

    def name(self):
        return 'CartoonDataset'
