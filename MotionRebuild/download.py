from django.http import FileResponse, Http404
import os

def download(request):
    if request.method == "GET":
        if 'file' in request.GET and request.GET['file']:
            filename = request.GET['file']
            if os.path.exists("./out_temp/" + filename):
                file = open("./out_temp/" + filename, 'rb')
                response = FileResponse(file, filename=filename.split("I")[0] + ".npy", as_attachment=True)
                response['Content-Type'] = 'application/octet-stream'
                return response
            else:
                raise Http404("找不到请求文件")
        else:
            raise Http404("找不到请求文件")
    else:
        raise Http404("请使用GET方法访问")