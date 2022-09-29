from typing import Union
from django.http import HttpResponse, JsonResponse, Http404, FileResponse
from django.core.files.uploadedfile import InMemoryUploadedFile

from .openpose_motion_rebuild.recognize import op_container
from .openpose_motion_rebuild.rebuild import rebuild2d, CapContainer, motion_adjust, RebuildFailException

import cv2
import numpy as np
import uuid
import datetime

recg = op_container("/home/luoxishuang/openpose/models/")

def ret_motion(request):
    if request.method == 'POST':
        imraw1 = None
        imraw2 = None

        save = True

        try:
            imraw1 = request.FILES.getlist('img1')[0]
            imraw2 = request.FILES.getlist('img2')[0]
        except IndexError:
            return JsonResponse({'status': 'error', 'msg': '请上传两张图片'})
        
        if imraw1 is None or imraw2 is None:
            return JsonResponse({'status': 'error', 'msg': '请上传两张图片'})

        data_raw = request.POST
        if 'img1' not in data_raw or 'img2' not in data_raw or 'adj' not in data_raw:
            return JsonResponse({'status': 'error', 'msg': '请确定焦平面角度'})
        
        angle1 = -1
        angle2 = -1
        angle_adjust = -1

        try:
            angle1 = float(data_raw['img1'])
            angle2 = float(data_raw['img2'])
            angle_adjust = float(data_raw['adj'])
        except ValueError:
            return JsonResponse({'status': 'error', 'msg': '请确定焦平面角度'})

        if angle1 == -1 or angle2 == -1:
            return JsonResponse({'status': 'error', 'msg': '请确定焦平面角度'})

        if 'save' in request.GET and request.GET['save'] and request.GET['save'] == 'False':
            save = False

        im1 = cv2.imdecode(np.frombuffer(imraw1.read(), np.uint8), flags=cv2.IMREAD_COLOR)
        im2 = cv2.imdecode(np.frombuffer(imraw2.read(), np.uint8), flags=cv2.IMREAD_COLOR)
        recg.setImage(im1)
        left = recg.getKeyPoint()
        recg.setImage(im2)
        right = recg.getKeyPoint()

        if left is None or right is None:
            return JsonResponse({'status': 'error', 'msg': '请确保两张图片都能被识别'})

        rebuilder = rebuild2d(angle1, angle2)  # 45, 315
        left_cap = CapContainer(angle2, left[0])  # 315
        right_cap = CapContainer(angle1, right[0])  # 45
        try:
            out = rebuilder.calc_depth(left_cap, right_cap)
        except AssertionError as e:
            return JsonResponse({'status': 'error', 'msg': e.__str__()})
        except RebuildFailException as e:
            return JsonResponse({'status': 'error', 'msg': e.__str__()})

        out = motion_adjust(out, angle_adjust)  # 45

        if save:
            outfile = None
            try:
                outfile = gene_filename()
                np.save("./out_temp/" + outfile, out)
            except FileNotFoundError:
                outfile = None
                pass
            
            if outfile is not None:
                return JsonResponse({'status': 'ok', 'msg': 'success', 'out': out.__repr__(), 'file': "./download?file=" + outfile})
        
        # no filename or save=False
        return JsonResponse({'status': 'ok', 'msg': 'success', 'out': out.__repr__()})
    else:
        raise Http404('请使用POST方法')

def gene_filename():
    return datetime.datetime.now().isoformat() + 'I' + uuid.uuid4().__str__() + ".npy"