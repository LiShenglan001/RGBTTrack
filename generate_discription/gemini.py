import copy
from scipy.stats import norm
import torch
from PIL import Image, ImageDraw
import google.generativeai as genai
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
from google.auth import default
from google.api_core.retry import Retry
from google.api_core.exceptions import TooManyRequests
import time
import os
import numpy as np
import ast
from torchvision import transforms
import math
import cv2 as cv
import torch.nn.functional as F

def load_gemini_model():
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPs_PROXY"] = "http://127.0.0.1:7890"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/cscv/Documents/lsl/SeqTrackv2/generate_discription/avid-catalyst-453707-h4-452f6de54af5.json"
    GOOGLE_API_KEY=os.getenv('AIzaSyBVEzCz_ME85lWHTNTfY6r2LfxmzY_fdFo')
    # GOOGLE_API_KEY = os.getenv('sk-C9Nd8e1cde644cb80ec5ae781b902a1019b35cbab94s0PuW')
    genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
    # creds, project = default()
    # print("Credentials:", creds)
    # print("Project:", project)
    for m in genai.list_models():
        print(m.name)
        print(m.supported_generation_methods)
    model = genai.GenerativeModel('gemini-1.5-flash') #'gemini-2.0-pro-exp', gemini-1.5-flash

    return model

def crop_image_CDF(image, bbox, output_img_path):
    width, height = image.size

    x_left, y_top, x_right, y_bottom = bbox
    w = (x_right - x_left)
    h = (y_bottom - y_top)
    x = x_left + w / 2
    y = y_top + h / 2

    # k_values = [2, 4, 6, 8]
    k_values = [24]

    cropped_images = []

    for k in k_values:
        sigma_x = width / k
        sigma_y = height / k

        def cdf_x(z):
            return norm.cdf(z, x, sigma_x)

        def cdf_y(z):
            return norm.cdf(z, y, sigma_y)

        prob_x = cdf_x(x + w / 2) - cdf_x(x - w / 2)
        prob_y = cdf_y(y + h / 2) - cdf_y(y - h / 2)

        w_new = w / prob_x
        h_new = h / prob_y

        left = max(0, int(x - w_new / 2))
        top = max(0, int(y - h_new / 2))
        right = min(width, int(x + w_new / 2))
        bottom = min(height, int(y + h_new / 2))

        # cropped_image = image[top:bottom, left:right]
        cropped_image = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_image)

    #     cv2.imwrite(f'cropped_k_{k}.jpg', cropped_image)
    #     cv2.imshow(f'k = {k}', cropped_image)
    #     cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    return cropped_images, k_values

def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image.crop((x_min, y_min, x_max, y_max))

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = np.array(im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :])
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded
        mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[
                0, 0]
        return im_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract = torch.tensor(box_extract)
    box_in = torch.tensor(box_in)
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * torch.tensor(resize_factor)
    box_out_wh = box_in[2:4] * torch.tensor(resize_factor)

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        #crop_sz[0] - 1, modified by chenxin from crop_sz[0],2022.7.15
        return box_out / (crop_sz[0]-1)
    else:
        return box_out

def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """


    crops_resize_factors = sample_target(frames, box_extract, search_area_factor, output_sz)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    # # find the bb location in the crop
    # '''Note that here we use normalized coord'''
    # box_crop = transform_image_to_crop(box_gt, box_extract, resize_factors, crop_sz, normalize=True)  # (x1,y1,w,h) list of tensors
    #
    # return frames_crop, box_crop, att_mask, masks_crop
    return crops_resize_factors

def crop_around_object(image,bbox_ori,output_img_path,search_area_factor=4.0,output_sz=256):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)
    x1, y1, x2, y2 = bbox_ori
    bbox = [x1, y1, x2-x1, y2-y1]
    image = image.permute(1, 2, 0)
    # im_show = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # cv2.rectangle(im_show, (int(bbox_ori[0]), int(bbox_ori[1])),
    #               (int(bbox_ori[2]), int(bbox_ori[3])), (0, 255, 0),
    #               3)
    # cv2.imshow('template_img_color', im_show)
    # cv2.waitKey()
    crops = jittered_center_crop(image, bbox, bbox, search_area_factor, output_sz)
    crops = np.array(crops * 256).squeeze().astype(np.uint8)
    # im_show = cv2.cvtColor(np.array(crops), cv2.COLOR_RGB2BGR)
    # cv2.imshow('template_img_color', im_show)
    # cv2.waitKey()
    return Image.fromarray(crops)

def to_markdown(text):
    text = text.replace('.', '*')
    return Markdown(textwrap.indent(text, '>', predicate=lambda _:True))

def generate_caption(image, model):
    # prompt = "Describe this image in one sentence within 20 words, without using any commas and without any additional text."
    # prompt = """
    # Describe the image in 40 words. Use short phrases. Follow this order:
    # 1. Main subject: appearance, shape, structure, texture, patterns
    # 2. Secondary elements: key features
    # 3. Spatial composition and perspective
    # """
    prompt = "Prompt: Write a concise and clear English sentence describing the object within the red rectangular frame, referencing its location, appearance, and type to ensure precise identification. The red frame is a prompt for you and should not be included in the generated description. The description should resemble something like: 'A red apple on the table.'"
    # response = model.generate_content([prompt, image], stream=False)
    # response.resolve()
    # description = response.text
    # description = description.strip()
    # description = description.replace('"', '')
    # description = description.replace("'", '')
    # description = description.replace("\n", '')
    # return description
    while True:
        try:
            response = model.generate_content([prompt, image], stream=False)
            response.resolve()
            description = response.text
            description = description.strip()
            description = description.replace('"', '')
            description = description.replace("'", '')
            description = description.replace("\n", '')
            # to_markdown(response.text)
            break
        except TooManyRequests as e:
            if e.code == 429:
                continue
    return description

def save_descriptions_to_file(video_folder_path, context_descriptions):

    try:
        language_file_path = os.path.join(video_folder_path, "gemini.txt")
        with open(language_file_path, "w") as f:
            for i in range(len(context_descriptions)):
                f.write(f"{context_descriptions[i]}\n")
        print(f"description has been saved: {language_file_path}")
    except Exception as e:
        print(f"failed to save description: {e}")

def read_bbox_from_init(video_folder_path):

    init_file_path = os.path.join(video_folder_path, "init.txt")
    with open(init_file_path, "r") as f:
        first_line = f.readline().strip()
        # first_line = [int(x) for x in first_line.split()]
        bbox = ast.literal_eval(first_line)  # [x, y, w, h]
        x, y, w, h = bbox
        return (x, y, x + w, y + h)  # (x_min, y_min, x_max, y_max)

def draw_box(img, box, output_image_path, outputname):
    draw = ImageDraw.Draw(img)
    # draw.rectangle(box, outline="red", width=2)
    output_image_path = os.path.join(output_image_path, outputname)
    img.save(output_image_path)

def describe_object_and_context(image_path, bbox, model, output_img_path):
    image = Image.open(image_path)
    # draw_box(copy.deepcopy(image), bbox, output_img_path, "gt-img.jpg")

    # object_image = crop_image(image, bbox)
    # draw_box(copy.deepcopy(object_image), bbox, output_img_path, "forground.jpg")
    # object_description = generate_caption(object_image, processor, model)

    # context_image = crop_around_object(image, bbox, output_img_path)
    # draw_box(copy.deepcopy(context_image), bbox, output_img_path, "context.jpg")
    # context_description = generate_caption(context_image, processor, model)

    context_images, k_values = crop_image_CDF(image, bbox, output_img_path)
    # draw_box(copy.deepcopy(context_images[0]), bbox, output_img_path, "context_0.5.jpg")
    # draw_box(copy.deepcopy(context_images[1]), bbox, output_img_path, "context_1.jpg")
    # draw_box(copy.deepcopy(context_images[2]), bbox, output_img_path, "context_2.jpg")
    context_descriptions = []
    for i in range(len(k_values)):
        context_descriptions.append(generate_caption(context_images[i], model))
    # context_description = " ".join(context_descriptions)
    context_descriptions.append(context_descriptions)
    return context_descriptions

def process_video_sequences(root_dir, model):
    n = 0
    for video_folder in os.listdir(root_dir):
        video_folder_path = os.path.join(root_dir, video_folder)
        if os.path.isdir(video_folder_path):
            print(f"processing: {video_folder}")
            description = os.path.join(video_folder_path, 'gemini.txt')
            if os.path.exists(description):
                n = n + 1
                print("description exists", n)
                continue
            visible_folder = os.path.join(video_folder_path, "visible")
            if os.path.exists(visible_folder):
                print(f"found: {visible_folder}")

                image_files = [f for f in os.listdir(visible_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:

                    image_files.sort()
                    # first_image = image_files[0]
                    # first_image_path = os.path.join(visible_folder, first_image)
                    first_image_path = os.path.join(video_folder_path, 'gt-img.jpg')
                    bbox = read_bbox_from_init(video_folder_path)
                    if bbox is None:
                        print("can't read box")
                        continue

                    context_descriptions = describe_object_and_context(first_image_path, bbox, model, video_folder_path)
                    # context_descriptions = optimize_description_with_chatgpt(context_descriptions)
                    if context_descriptions:
                        save_descriptions_to_file(video_folder_path, context_descriptions)
                    else:
                        print("can't generate description")
                else:
                    print("visible folder has no images")
            else:
                print(f"can't find visible folder: {visible_folder}")

def main():
    root_directory = "/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/trainingset_ori"
    model = load_gemini_model()
    if model is None:
        return

    process_video_sequences(root_directory, model)

if __name__ == "__main__":
    main()
