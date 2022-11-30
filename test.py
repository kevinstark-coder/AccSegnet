import time
import os
import sublist
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from util.visualizer import Visualizer
import torchnet.meter as meter


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    seg_output_dir = opt.test_seg_output_dir
    '''test_img_list_file = opt.test_img_list_file
    opt.imglist_testB = sublist.dir2list(opt.test_B_dir, test_img_list_file)'''
    dice = meter.AverageValueMeter()
    dice.reset()
    assd = meter.AverageValueMeter()
    assd.reset()
    

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        #model.save_2D_image('/media/ubuntu/name/acc_result/predict')
        #model.save_gt_image('/media/ubuntu/name/acc_result/target')
        model.save_image('/media/ubuntu/name/acc_result/image')
        dice.add(model.our_validdice(), n=1)
        assd.add(model.our_validassd(), n=1)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('processing image... %s' % img_path)
        visualizer.save_seg_images_to_dir(seg_output_dir, visuals, img_path)
    
    Dice_mean = dice.value()[0]
    Dice_var = dice.value()[1]
    Assd_mean = assd.value()[0]
    Assd_var = assd.value()[1]
    print('Dice Coeff: {},{}'.format(Dice_mean, Dice_var))
    print('ASSD: {},{}'.format(Assd_mean, Assd_var))