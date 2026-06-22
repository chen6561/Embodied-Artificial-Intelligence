import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ====================== 全局配置 ======================
# 数据集根目录（存放你下载的zip文件的文件夹）
DATASET_ROOT = "./middlebury_dataset"
# 要处理的场景列表
SCENES = ["cones", "teddy"]
# 要处理的分辨率列表
RESOLUTIONS = ["Q", "H", "F"]  # Q=四分之一, H=半, F=全
# SGBM算法参数（工业级调优版）
SGBM_PARAMS = {
    "minDisparity": 0,
    "numDisparities": 64,  # 必须是16的倍数，根据场景调整
    "blockSize": 5,        # 必须是奇数，3~11之间
    "P1": 8 * 3 * 5 ** 2,
    "P2": 32 * 3 * 5 ** 2,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 2,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
}

# ====================== 1. 自动解压数据集 ======================
def unzip_dataset():
    """自动解压当前目录下所有Middlebury zip文件"""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    for file in os.listdir("."):
        if file.endswith(".zip") and ("cones" in file or "teddy" in file):
            # 解析场景和分辨率
            parts = file.split("-")
            scene = parts[0][:-1]  # conesQ → cones
            res = parts[0][-1]    # conesQ → Q
            
            extract_path = os.path.join(DATASET_ROOT, f"{scene}_{res}")
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            print(f"解压完成: {file} → {extract_path}")

# ====================== 2. 加载单组数据 ======================
def load_scene_data(scene, resolution):
    """
    加载指定场景和分辨率的双目数据
    返回: left_img, right_img, gt_disparity
    """
    data_path = os.path.join(DATASET_ROOT, f"{scene}_{resolution}")
    
    # 读取左右图像（ppm格式）
    left_img = np.array(Image.open(os.path.join(data_path, "im2.ppm")))
    right_img = np.array(Image.open(os.path.join(data_path, "im6.ppm")))
    
    # 转换为灰度图（立体匹配需要灰度图）
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    
    # 读取真实视差图（16位PNG，低4位是小数部分）
    gt_disparity = np.array(Image.open(os.path.join(data_path, "disp2.pgm")))
    gt_disparity = gt_disparity.astype(np.float32) / 16.0  # 转换为真实视差
    
    # 无效点标记为0（遮挡区域）
    gt_disparity[gt_disparity < 0] = 0
    
    return left_img, right_img, left_gray, right_gray, gt_disparity

# ====================== 3. 立体匹配计算视差图 ======================
def compute_disparity(left_gray, right_gray, params):
    """使用SGBM算法计算视差图"""
    sgbm = cv2.StereoSGBM_create(**params)
    disparity = sgbm.compute(left_gray, right_gray)
    disparity = disparity.astype(np.float32) / 16.0  # 转换为真实视差
    
    # 去除负视差和超出范围的视差
    disparity[disparity < params["minDisparity"]] = 0
    disparity[disparity > params["minDisparity"] + params["numDisparities"]] = 0
    
    return disparity

# ====================== 4. 精度评估 ======================
def evaluate_disparity(pred_disparity, gt_disparity, threshold=1.0):
    """
    计算视差图的精度指标
    返回: MAE, RMSE, BadPixelRatio
    """
    # 只计算有真实视差的区域
    mask = gt_disparity > 0
    pred_valid = pred_disparity[mask]
    gt_valid = gt_disparity[mask]
    
    if len(pred_valid) == 0:
        return 0, 0, 100
    
    error = np.abs(pred_valid - gt_valid)
    
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    bad_pixel_ratio = np.mean(error > threshold) * 100
    
    return mae, rmse, bad_pixel_ratio

# ====================== 5. 结果可视化 ======================
def visualize_results(scene, resolution, left_img, right_img, gt_disparity, pred_disparity, metrics):
    """可视化结果并保存"""
    mae, rmse, bpr = metrics
    
    plt.figure(figsize=(20, 12))
    
    # 左图像
    plt.subplot(2, 3, 1)
    plt.imshow(left_img)
    plt.title(f"{scene} ({resolution}) - 左图像", fontsize=14)
    plt.axis('off')
    
    # 右图像
    plt.subplot(2, 3, 2)
    plt.imshow(right_img)
    plt.title("右图像", fontsize=14)
    plt.axis('off')
    
    # 真实视差图
    plt.subplot(2, 3, 3)
    plt.imshow(gt_disparity, cmap='jet')
    plt.title("真实视差图", fontsize=14)
    plt.colorbar(label='视差 (像素)')
    plt.axis('off')
    
    # 计算的视差图
    plt.subplot(2, 3, 4)
    plt.imshow(pred_disparity, cmap='jet')
    plt.title(f"SGBM计算视差图\nMAE={mae:.2f}像素", fontsize=14)
    plt.colorbar(label='视差 (像素)')
    plt.axis('off')
    
    # 误差图
    plt.subplot(2, 3, 5)
    error_map = np.abs(pred_disparity - gt_disparity)
    error_map[gt_disparity == 0] = 0
    plt.imshow(error_map, cmap='hot', vmin=0, vmax=5)
    plt.title(f"误差图\nRMSE={rmse:.2f}像素, 错误率={bpr:.2f}%", fontsize=14)
    plt.colorbar(label='误差 (像素)')
    plt.axis('off')
    
    # 误差直方图
    plt.subplot(2, 3, 6)
    mask = gt_disparity > 0
    errors = np.abs(pred_disparity[mask] - gt_disparity[mask])
    plt.hist(errors, bins=50, range=(0, 5))
    plt.title("误差分布直方图", fontsize=14)
    plt.xlabel("误差 (像素)")
    plt.ylabel("像素数量")
    
    plt.tight_layout()
    
    # 保存结果
    save_path = f"{scene}_{resolution}_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"结果已保存到: {save_path}")
    
    plt.show()

# ====================== 6. 主函数：批量处理所有场景 ======================
def main():
    # 第一步：自动解压所有zip文件
    print("正在解压数据集...")
    unzip_dataset()
    
    # 第二步：批量处理每个场景和分辨率
    for scene in SCENES:
        for res in RESOLUTIONS:
            try:
                print(f"\n{'='*50}")
                print(f"正在处理: {scene} ({res}分辨率)")
                print('='*50)
                
                # 加载数据
                left_img, right_img, left_gray, right_gray, gt_disparity = load_scene_data(scene, res)
                print(f"图像尺寸: {left_img.shape[1]}×{left_img.shape[0]}")
                print(f"最大真实视差: {np.max(gt_disparity):.2f}像素")
                
                # 根据分辨率调整SGBM参数
                params = SGBM_PARAMS.copy()
                if res == "F":
                    params["numDisparities"] = 128  # 全分辨率视差范围更大
                    params["blockSize"] = 7
                elif res == "H":
                    params["numDisparities"] = 64
                    params["blockSize"] = 5
                else:  # Q
                    params["numDisparities"] = 32
                    params["blockSize"] = 3
                
                # 计算视差图
                pred_disparity = compute_disparity(left_gray, right_gray, params)
                
                # 评估精度
                metrics = evaluate_disparity(pred_disparity, gt_disparity)
                mae, rmse, bpr = metrics
                print(f"平均绝对误差(MAE): {mae:.2f}像素")
                print(f"均方根误差(RMSE): {rmse:.2f}像素")
                print(f"错误率(>1像素): {bpr:.2f}%")
                
                # 可视化结果
                visualize_results(scene, res, left_img, right_img, gt_disparity, pred_disparity, metrics)
                
            except Exception as e:
                print(f"处理 {scene}_{res} 失败: {e}")
                continue

if __name__ == "__main__":
    main()