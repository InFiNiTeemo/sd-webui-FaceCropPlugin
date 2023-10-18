from fastapi import FastAPI, Body

from modules.api.models import *
from modules.api import api
import gradio as gr

import cv2
from PIL import Image
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def crop_face_api(_: gr.Blocks, app: FastAPI):
    @app.post("/hoxi_crop_face")
    async def face_crop(
        input_image: str = Body("", title='crop_face input image'),
        model: str = Body("buffalo_l", title='face Recognition model'), 
    ):
        print('start')
        input_image = api.decode_base64_to_image(input_image)

        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        img = np.array(input_image)[:,:,:3]
        faces = app.get(img)
        bbox = faces[0]['bbox']
        x1,y1,x2,y2 = bbox
        w,h = x2-x1, y2-y1
        img = np.array(input_image)[:,:,:4]
        _w,_h,_ = img.shape

        crop_box = np.array([max(0,x1-w),max(0,y1-h),min(x2+w,_w),min(y2+h,_h)])
        image = img[int(crop_box[1]):int(crop_box[3]), int(crop_box[0]):int(crop_box[2]), :]
        face_h,face_w,C = image.shape
        if face_h>face_w:
            tar_h = 200
            tar_w = int(200/(face_h/face_w))
            image = cv2.resize(image,(tar_w,tar_h))
            pad_l  = (200-tar_w)//2
            pad_r = 200 - pad_l - tar_w
            image = np.hstack([np.ones([tar_h,pad_l,C]), image,np.ones([tar_h,pad_r,C])])
        else:
            tar_h = int(200/(face_w/face_h))
            tar_w = 200
            image = cv2.resize(image,(tar_w,tar_h))
            pad_l  = (200-tar_h)//2
            pad_r = 200 - pad_l - tar_h
            image = np.vstack([np.ones([pad_l, tar_w, C]), image,np.ones([pad_r,tar_w,C])])

        image = Image.fromarray(image.astype('uint8'))

            # faces[0]['bbox'] = np.array([max(0,x1-w),max(0,y1-h),min(x2+w,_w),min(y2+h,_h)])
            # rimg = app.draw_on(img, faces)
            # cv2.imwrite("/root/autodl-tmp/xiaoyu/a.jpg",rimg)

        return {"images": [api.encode_pil_to_base64(image).decode("utf-8")]}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(crop_face_api)
except:
    pass