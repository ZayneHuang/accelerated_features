import os
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import time

from modules.xfeat import XFeat

# 强制使用 CPU，如果希望使用 GPU，请注释掉下面一行
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化 XFeat 模块
xfeat = XFeat(weights='./weights/xfeat.pt', init_lighter_glue=True)

app = FastAPI()


class ImageData(BaseModel):
    image1: str  # base64编码的图像数据
    image2: str  # base64编码的图像数据


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    将 base64 编码的字符串转换为 RGB 格式的 numpy 数组图像。
    如果存在 "data:image/xxx;base64," 前缀，则会自动去除。
    """
    try:
        # 去除可能存在的前缀信息
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图像解码失败")
        # 将 OpenCV 默认的 BGR 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise ValueError("无效的图像数据") from e


def to_serializable(val):
    """
    递归将 numpy 数组、字典、列表等转换为 JSON 可序列化的数据类型。
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, torch.Tensor):
        # 先 detach 再转换为 CPU 上的列表
        return val.detach().cpu().tolist()
    elif isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(item) for item in val]
    else:
        return val


@app.post("/process")
async def process_images(data: ImageData):
    t1 = time.time()
    try:
        # 将 base64 字符串转换为图像
        im1 = base64_to_image(data.image1)
        im2 = base64_to_image(data.image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    t2 = time.time()
    print(f"解析图片时长: {t2 - t1}")

    # 特征提取：取返回结果中的第一个
    output0 = xfeat.detectAndCompute(im1, top_k=4096)[0]
    output1 = xfeat.detectAndCompute(im2, top_k=4096)[0]
    
    t3 = time.time()

    print(f"特征提取时长: {t3 - t2}")

    # 更新图像尺寸（必需）
    output0.update({'image_size': (im1.shape[1], im1.shape[0])})
    output1.update({'image_size': (im2.shape[1], im2.shape[0])})
    

    # 使用 lighter glue 进行特征匹配
    mkpts_0, mkpts_1, idxs = xfeat.match_lighterglue(output0, output1)

    t4 = time.time()

    print(f"图像匹配时长: {t4 - t3}")

    # 转换所有结果为 JSON 可序列化的数据
    result = {
        "output0": to_serializable(output0),
        "output1": to_serializable(output1),
        "mkpts_0": to_serializable(mkpts_0),
        "mkpts_1": to_serializable(mkpts_1),
        "idxs": to_serializable(idxs)
    }

    t5 = time.time()

    print(f"结果序列化时长: {t5 - t4}")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xfeat_lg_serve:app", host="0.0.0.0", port=8000, reload=True)
