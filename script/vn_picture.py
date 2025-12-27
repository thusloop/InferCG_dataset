import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# 尝试导入 venn 库
try:
    from venn import venn
except ImportError:
    print("请先安装 venn 库以生成4集合韦恩图: pip install venn")
    venn = None

def calculate_metrics(true_edges, predicted_edges):
    """
    计算精准率、召回率，并找出真实中存在但预测中不存在的边，以及预测中存在但真实中不存在的边。
    """
    tp = true_edges & predicted_edges
    fn = true_edges - predicted_edges
    fp = predicted_edges - true_edges

    precision = len(tp) / len(predicted_edges) if predicted_edges else 0
    recall = len(tp) / len(true_edges) if true_edges else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1, fn, fp

def load_json(name, file_path, replace_list=[], skip_list=[], pre_clean=[], all_or_EA=1):
    """
    加载 JSON 文件并将边关系转换为集合形式。
    """
    if not os.path.exists(file_path):
        # print(f"Warning: File not found {file_path}")
        return set()

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = set()
    for source, targets in data.items():
        if source.split('.')[0] in skip_list:
            continue
        for r in replace_list:
            source = source.replace(r[0], r[1])
        
        if all_or_EA == 0:
            if not source.startswith('<') and not source.startswith(name):
                continue

        for target in targets:
            if "error" in target.lower():
                continue
            if target.split('.')[0] in skip_list:
                continue
            for r in replace_list:
                target = target.replace(r[0], r[1])
            
            if all_or_EA == 0:
                if not target.startswith('<') and not target.startswith(name):
                    continue   
            
            edges.add((source, target))

    return edges

def main(name, replace_list, skip_list, pre_clean=[], precision_write=0, recall_write=0, Ae_or_pycg=1, all_or_EA=1):
    """
    主函数，加载 JSON 文件，计算指标。
    返回: P, R, F1, 真实边集合, 预测边集合
    """
    true_json_path = "../ground-truth-cgs/{}.json".format(name)
    
    if Ae_or_pycg == 1:
        predicted_json_path = "../Ae_data/{}.json".format(name)
    elif Ae_or_pycg == 0:
        predicted_json_path = "../PyCG_data/{}.json".format(name)
    elif Ae_or_pycg == 2:
        predicted_json_path = "../Depends_data/{}.json".format(name)
    elif Ae_or_pycg == 3:
        predicted_json_path = "../coarseCG_data/{}.json".format(name)
    
    # 加载边集合
    true_edges = load_json(name, true_json_path, replace_list, skip_list, pre_clean, all_or_EA)
    predicted_edges = load_json(name, predicted_json_path, replace_list, skip_list, pre_clean, all_or_EA)

    # 计算指标
    precision, recall, f1, fn, fp = calculate_metrics(true_edges, predicted_edges)
    
    return precision*100, recall*100, f1*100, true_edges, predicted_edges

def draw_venn(sets_dict, title, save_path):
    """
    绘制并保存 Venn 图 (去空白版)。
    """
    if venn is None:
        return

    # 1. 调整画布大小，稍微扁一点可能更适合4集合的形状
    plt.figure(figsize=(10, 8))
    
    # 2. 绘制 Venn 图
    # legend_loc="best" 让库自动寻找最佳位置，或者手动指定 "upper left"
    venn(sets_dict, fontsize=12, legend_loc="upper left")
    
    # 3. 设置标题，y参数控制标题高度，防止标题离图太远
    plt.title(title, fontsize=16, y=1.02)
    
    # 4. 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 5. 核心修改：bbox_inches='tight' 自动裁剪空白
    # pad_inches=0.1 给边缘留一点点呼吸空间，不要切到了字
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.close()
    print(f"Total Venn diagram saved to {save_path}")

# --- 全局变量初始化 ---

# project_name_list_1 = ["asciinema", "autojump"] 
# project_name_list_2 = ['sqlparse', 'sshtunnel', 'textrank4zh'] 
project_name_list_1 = ["asciinema","autojump","fabric","face_classification","Sublist3r"]
project_name_list_2 = ['bpytop','furl','rich_cli','sqlparse','sshtunnel','textrank4zh']
skip_list = ['test', 'tests', 'setup', 'uninstall', 'install', 'setup.py']
replace_list = [
    ('.__init__', ''),
    ('<**PyList**>', '<list>'),
    ("<**PyStr**>", "<str>"),
    ("<**PyDict**>", "<map>"),
    ("<**PySet**>", "<set>"),
    ("<**PyTuple**>", "<tuple>"),
    ("<**PyNum**>", "<num>"),
    ("<**PyBool**>", "<bool>"),
    ('Socket', 'socket'),
    ('<builtin>', '<builtin>'),
    ('Socket', 'socket'),
    ('invoke.config.Config._set', 'invoke.config.DataProxy._set'),
    ('invoke.Context._set', 'invoke.config.DataProxy._set'),
    ('fabric.util.debug', 'util.debug'),
    ('invoke.terminals.pty_size', 'invoke.pty_size'),
    ('invoke.context.Context._run', "invoke.Context._run"),
    ('invoke.context.Context._sudo', "invoke.Context._sudo"),
    ('invoke.parser.argument.Argument', "invoke.Argument"),
    ('invoke.program.Program.core_args', "invoke.Program.core_args"),
    ('invoke.collection.Collection', "invoke.Collection"),
    ('invoke.program.Program.load_collection', "invoke.Program.load_collection"),
    ('invoke.program.Program.no_tasks_given', "invoke.Program.no_tasks_given"),
    ('invoke.program.Program.update_config', "invoke.Program.update_config"),
    ('invoke.collection.Collection', "invoke.Collection"),
    ('re.compile.findall', 're.findall'),
    ('requests.sessions.Session.get', 'requests.Session.get'),
    ('argparse.ArgumentParser.add_argument', 'argparse.add_argument'),
    ('argparse.ArgumentParser.parse_args', 'argparse.parse_args'),
    ('threading.Thread.start', 'threading.start')
]

# 用于存储所有项目的累计指标 (Depends 已移除)
stats = {
    'Ae': {'p': 0, 'r': 0, 'f1': 0},
    'PyCG': {'p': 0, 'r': 0, 'f1': 0},
    'coarseCG': {'p': 0, 'r': 0, 'f1': 0}
}

# 全局集合
# all_gt: 所有真实的边
# 格式: (project_name, source_node, target_node)
all_gt = set()
all_tp_Ae = set()
all_tp_PyCG = set()
all_tp_coarseCG = set()

def process_project_and_accumulate(name, replace_list, skip_list, all_or_EA):
    """
    处理单个项目：计算指标，并将正确识别的边以及真实边加入全局集合
    """
    print(f"Processing {name}...")
    
    # 1. Ae (InferCG Full)
    p, r, f1, true_edges, pred_ae = main(name, replace_list, skip_list, precision_write=0, recall_write=0, Ae_or_pycg=1, all_or_EA=all_or_EA)
    stats['Ae']['p'] += p; stats['Ae']['r'] += r; stats['Ae']['f1'] += f1
    
    # 将该项目的真实边加入全局 GT 集合
    for edge in true_edges:
        all_gt.add((name, edge[0], edge[1]))

    # Ae TP (正确预测的边)
    tp_ae = pred_ae
    for edge in tp_ae:
        all_tp_Ae.add((name, edge[0], edge[1]))
    
    # 2. PyCG
    p, r, f1, _, pred_pycg = main(name, replace_list, skip_list, precision_write=0, recall_write=0, Ae_or_pycg=0, all_or_EA=all_or_EA)
    stats['PyCG']['p'] += p; stats['PyCG']['r'] += r; stats['PyCG']['f1'] += f1
    tp_pycg = pred_pycg
    for edge in tp_pycg:
        all_tp_PyCG.add((name, edge[0], edge[1]))

    # 3. CoarseCG (InferCG Coarse)
    p, r, f1, _, pred_coarse = main(name, replace_list, skip_list, precision_write=0, recall_write=0, Ae_or_pycg=3, all_or_EA=all_or_EA)
    stats['coarseCG']['p'] += p; stats['coarseCG']['r'] += r; stats['coarseCG']['f1'] += f1
    tp_coarse = pred_coarse
    for edge in tp_coarse:
        all_tp_coarseCG.add((name, edge[0], edge[1]))

# --- 执行循环 ---

pre_clean = []

# List 1
for name in project_name_list_1:
    if name == "autojump":
        replace_list.append(('bin.', ''))
    elif name == "face_classification":
        replace_list.append(('src.', ''))
        
    process_project_and_accumulate(name, replace_list, skip_list, all_or_EA=1)
    
    if name == "autojump":
        replace_list.pop()
    elif name == "face_classification":
        replace_list.pop()

# List 2
for name in project_name_list_2:
    process_project_and_accumulate(name, replace_list, skip_list, all_or_EA=0)


# --- 绘制总 Venn 图 ---

# 将全局集合放入字典
# 注意：PyCG, coarseCG, Ae 的集合都是 GT 的子集（因为我们只取了 TP）
# 这将展示每个工具覆盖了 GT 的哪一部分
venn_data = {
    "Ground Truth": all_gt,
    "InferCG(DeepSeek)": all_tp_Ae,
    "PyCG": all_tp_PyCG,
    "Coarse-Grained Static Analysis": all_tp_coarseCG
}

# 绘制并保存
save_path = "../venn_diagrams/total_coverage_venn.png"
draw_venn(venn_data, title="", save_path=save_path)


# --- 打印平均指标结果 ---
total_projects = len(project_name_list_1) + len(project_name_list_2)
print("\n=== Average Metrics Results ===")
for tool, vals in stats.items():
    print(f"{tool} average pre={vals['p']/total_projects:.3f}")
    print(f"{tool} average rec={vals['r']/total_projects:.3f}")
    print(f"{tool} average F1={vals['f1']/total_projects:.3f}")

print("\n=== Venn Diagram Data Info ===")
print(f"Total True Edges (Ground Truth): {len(all_gt)}")
print(f"Total Correct Edges by Ae: {len(all_tp_Ae)}")
print(f"Total Correct Edges by PyCG: {len(all_tp_PyCG)}")
print(f"Total Correct Edges by coarseCG: {len(all_tp_coarseCG)}")
print(f"Venn diagram saved to {save_path}")