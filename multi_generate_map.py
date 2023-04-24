import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel


model_name_list = ['SwinIR@Base', 'RCAN@Base', 'HAN@Base', 'EDSR@Base']
# model_name = 'SwinIR@Base'

input_dic = '/data1/zyli/OST_selected_GTmod32/'
map_output_dic = 'OST'

# Load test image
window_size = 32  # Define windoes_size of D


def generate_map(input_img_path, model, output_dic):
    img_lr, img_hr = prepare_images(input_img_path)  # Change this image name
    eps = 1e-7

    # img_lr.save('lr/7_lr.png')
    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    tensor_lr += torch.full(tensor_lr.size(), eps)
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    plt.imshow(cv2_hr)

    w = 128  # The x coordinate of your select patch, 125 as an example
    h = 144  # The y coordinate of your select patch, 160 as an example
             # And check the red box
             # Is your selected patch this one? If not, adjust the `w` and `h`.


    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)
    position_pil

    sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         blend_abs_and_input,
         blend_kde_and_input,
         Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
    )
    img_name = img_path.split('/')[-1]
    pil.save(os.path.join(output_dic, img_name+'_'+model_name+'.bmp'))

    gini_index = gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    print(f"The DI of this case is {diffusion_index}")


img_paths = []
for name in os.listdir(input_dic):
    img_paths.append(os.path.join(input_dic, name))###这里的'.tif'可以换成任意的文件后缀
for model_name in model_name_list:
    model = load_model(model_name)  # You can Change the model name to load different model
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        print(img_path)
        generate_map(img_path, model, map_output_dic)