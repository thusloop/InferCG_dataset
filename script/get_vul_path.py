import os

def list_fx_dirs(root_dir):
    """
    遍历 root_dir 下所有 proX/fY 文件夹，只输出 fY 层，不进入更深层
    """
    result = []

    # 遍历一级子目录（pro1, pro2, ...）
    for pro in os.listdir(root_dir):
        pro_path = os.path.join(root_dir, pro)
        if not os.path.isdir(pro_path):
            continue

        # 遍历二级子目录（f1, f2, ...）
        for fx in os.listdir(pro_path):
            fx_path = os.path.join(pro_path, fx)
            if os.path.isdir(fx_path):
                # 只记录 fX 层，不再深入
                result.append(os.path.abspath(fx_path))

    return result


if __name__ == "__main__":
    root = r"E:\001_some_AI_code\003_AutoExtension\STAR\vulnerable_fun"   # 修改成你的根目录
    fx_dirs = list_fx_dirs(root)

    print("所有 fX 子目录路径如下：")
    print("[")
    for d in fx_dirs:
        print(f"r\"{d}\",")
    print("]")
