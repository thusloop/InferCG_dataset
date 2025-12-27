import csv
import re
import json

project_name_list_1 = ["asciinema","autojump","fabric","face_classification","Sublist3r"]
#project_name_list_1 = ["asciinema","autojump"] #测试集
project_name_list_1 = ["fabric","face_classification","Sublist3r"] #训练集
project_name_list_2 = ['bpytop','furl','rich_cli','sqlparse','sshtunnel','textrank4zh']
#project_name_list_2 = ['sqlparse','sshtunnel','textrank4zh'] #测试集
project_name_list_2 = ['bpytop','furl','rich_cli'] #训练集
project_list_id=[1,3,4,7,8,11,12,13,14,15,16,22,23,28,29,32,33,35,36,38,45,48,53,56,58]


def calculate_metrics(true_edges, predicted_edges):
    """
    计算精准率、召回率，并找出真实中存在但预测中不存在的边，以及预测中存在但真实中不存在的边。
    """
    # 真正例（TP）：真实和预测都存在的边
    tp = true_edges & predicted_edges

    # 真实中存在，但预测中不存在的边（FN）
    fn = true_edges - predicted_edges

    # 预测中存在，但真实中不存在的边（FP）
    fp = predicted_edges - true_edges

    # 计算精准率和召回率
    # print("tp={}".format(len(tp)))
    # print("predicted_edges={}".format(len(predicted_edges)))
    # print("true_edges={}".format(len(true_edges)))
    precision = len(tp) / len(predicted_edges) if predicted_edges else 0
    recall = len(tp) / len(true_edges) if true_edges else 0

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1, fn, fp


import json
from collections import defaultdict
def merge_list_values(pairs):
    """合并重复键的列表值"""
    merged = defaultdict(list)
    for key, value in pairs:
        # 确保值都是列表（如果不是，转换为列表）
        if not isinstance(value, list):
            value = [value]
        merged[key].extend(value)  # 合并列表内容
    
    return dict(merged)
def load_json_1(name, file_path, replace_list=[], skip_list=[], pre_clean=[],all_or_EA = 1):
    """
    加载 JSON 文件并将边关系转换为集合形式，便于比较。
    返回边集合，格式为 {(source, target), ...}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)



    edges = set()
    for source, targets in data.items():
        if source.split('.')[0] in skip_list:
            continue
        # if source.split('.')[0] in pre_clean:
        #     source = '.'.join(source.split('.')[1:])
        # if 'lambda' in source:
        #     continue
        for r in replace_list:
            source = source.replace(r[0],r[1])
        # if source in replace_map:
        #     source = replace_map[source]
        # if not source.startswith('<') and not source.startswith('fabric') and not source.startswith('integration'):
        #         continue
        if all_or_EA == 0:
            if not source.startswith('<') and not source.startswith(name):
                continue
        #source = source.split('.')[0] + source.split('.')[-1]
        for target in targets:
            # if target.startswith('<'):
            #     continue
            if "error" in target.lower():
                continue
            # if 'lambda' in target:
            #     continue
            if target.split('.')[0] in skip_list:
                continue
            # if target.split('.')[0] in pre_clean:
            #     target = '.'.join(target.split('.')[1:])
            for r in replace_list:
                target = target.replace(r[0],r[1])

            # if target in replace_map:
            #     target = replace_map[target] 
            # if not target.startswith('<') and not target.startswith('fabric') and not target.startswith('integration'):
            #     continue   
            if all_or_EA == 0:
                if not target.startswith('<') and not target.startswith(name):
                    continue   
            #target = target.split('.')[0] + '.' + target.split('.')[-1]
            edges.add((source, target))

    return edges

def main(name, replace_list, skip_list, pre_clean=[], precision_write = 0, recall_write = 0, Ae_or_pycg = 1, all_or_EA = 1):
    """
    主函数，加载 JSON 文件，计算指标并输出结果。
    """
    true_json_path = "../ground-truth-cgs/{}.json".format(name)  # 真实边的 JSON 文件路径
    #true_json_path = "../PyCG_data/{}.json".format(name)
    if Ae_or_pycg == 1:
        #print("{} Ae:".format(name))
        predicted_json_path = "../Ae_data/{}.json".format(name)  # 预测边的 JSON 文件路径
    elif Ae_or_pycg == 0 :
        #print("{} PyCG:".format(name))
        predicted_json_path = "../PyCG_data/{}.json".format(name)  # 预测边的 JSON 文件路径
    elif Ae_or_pycg == 2 :
        #print("{} Depends:".format(name))
        predicted_json_path = "../Depends_data/{}.json".format(name)  # 预测边的 JSON 文件路径
    
    # 加载边集合
    true_edges = load_json_1(name, true_json_path, replace_list,skip_list, pre_clean, all_or_EA)
    predicted_edges = load_json_1(name, predicted_json_path, replace_list, skip_list, pre_clean, all_or_EA)

    # 计算指标
    precision, recall, f1, fn, fp = calculate_metrics(true_edges, predicted_edges)
    precision = precision*100
    recall = recall*100
    f1 = f1*100
    # 输出结果
    
    #print(f"Precision (精准率): {precision:.3f}")
    #print(f"Recall (召回率): {recall:.3f}")
    #print(f"F1: {f1:.3f}\n\n")

    if recall_write:
        #召回率
        print("Edges in true JSON but not in predicted JSON (FN):")
        for edge in sorted(fn):
            print(edge)

    if precision_write:
        # 精准率
        print("\nEdges in predicted JSON but not in true JSON (FP):")
        for edge in sorted(fp):
            print(edge)
    return precision, recall, f1


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_percentage(text):
    matches = re.findall(r'(?<![-\d])(?:100|[1-9]?\d(?:\.\d)?)%(?!\d)', text)
    percentages = [float(match.rstrip('%')) for match in matches]  # 去掉 '%' 并转换为浮点数
    return min(percentages) if percentages else None  # 返回最小值，如果为空则返回 None

def solve(name,input_file1,input_file2, rate_threshold, replace_list, skip_list, pre_clean, all_or_EA, op = 'merged'): 

    if op == 'merged':
        #print("merged")
        # 有QWEN
        true_json_path = "../ground-truth-cgs/{}.json".format(name)
        true_edges = load_json_1(name, true_json_path, replace_list, skip_list, pre_clean, all_or_EA)
        project_data = {}
        project_json_path = {}

        # for name in project_name_list_1 + project_name_list_2:
        #     project_json_path[name] = "../PyCG_data/{}.json".format(name)
        #     # project_data[name] = load_json(project_json_path[name])
        #     project_data[name] = {}
        #     if name in project_name_list_2:
        #         project_json_path[name] = "../file2class_data/{}.json".format(name)
        #         project_data[name] = load_json(project_json_path[name])
        

        caller_candidates = []
        project_name_idx = []
        with open(input_file1, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[5] != name:
                    continue
                caller = row[3]
                callee = row[4]
                caller_candidates.append({"caller":caller,"callee":callee})
                project_name_idx.append(row[5])

        result = [0] * len(caller_candidates)
        
        tp, fn, fp, tn = 0,0,0,0
        with open(input_file2, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # 跳过标题行
            next(reader)
            for i,row in enumerate(reader):
                if row[9] != name:
                    continue
                custom_id = i
                #print(custom_id)
                content = row[7]
                #project_name_idx.append(row[9])
                #project_name_idx[custom_id] = row[9]
                precent = extract_percentage(content)

                if  precent >= rate_threshold:
                    if (caller_candidates[custom_id]["caller"], caller_candidates[custom_id]["callee"]) in true_edges:
                        result[custom_id] += 1
                        tp += 1
                    else:
                        result[custom_id] += 0
                        fp += 1
                else :
                    if (caller_candidates[custom_id]["caller"], caller_candidates[custom_id]["callee"]) in true_edges:
                        result[custom_id] += 0
                        fn += 1
                    else:
                        result[custom_id] += 0
                        tn += 1


    else :


        project_data = {}
        project_json_path = {}
        
        project_json_path[name] = "../PyCG_data/{}.json".format(name)
        project_data[name] = load_json(project_json_path[name])
        project_data[name] = {}
        if name in project_name_list_2:
            project_json_path[name] = "../file2class_data/{}.json".format(name)
            project_data[name] = load_json(project_json_path[name])

        caller_candidates = []
        with open(input_file1, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                caller = row[3]
                callee = row[4]
                caller_candidates.append({"caller":caller,"callee":callee})

        result = [0] * len(caller_candidates)
        project_name_idx = []
        with open(input_file2, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # 跳过标题行
            next(reader)
            for i,row in enumerate(reader):
                custom_id = int(row[0])
                content = row[7]
                project_name_idx.append(row[9])
                precent = extract_percentage(content)
                if precent >= rate_threshold:
                    result[i] += 1
                

        for i, call in enumerate(caller_candidates):

            caller, candidate = call["caller"], call["callee"]
            yes_total = result[i]
            name = project_name_idx[i]
            if yes_total:
                if caller in project_data[name]:
                    project_data[name][caller].append(candidate)  # 将 candidate 添加到原列表
                    # 去重
                    project_data[name][caller] = list(set(project_data[name][caller]))
                else:
                    project_data[name][caller] = []
                    project_data[name][caller].append(candidate)  # 如果 caller 不存在，直接插入


        filename_with_extension = project_json_path[name].split('/')[-1]
        with open("../Ae_data/{}".format(filename_with_extension), "w", encoding="utf-8") as file:
            json.dump(project_data[name], file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    pre_clean = []
    skip_list = ['test','tests','setup','uninstall','install','setup.py']
    replace_list = [
        ('.__init__',''),
        ('<**PyList**>','<list>'),
        ("<**PyStr**>","<str>"),
        ("<**PyDict**>","<map>"),
        ("<**PySet**>","<set>"),
        ("<**PyTuple**>","<tuple>"),
        ("<**PyNum**>","<num>"),
        ("<**PyBool**>","<bool>"),
        ('Socket','socket'),
        ('<builtin>','<builtin>'),
        ('Socket','socket'),
        ('invoke.config.Config._set','invoke.config.DataProxy._set'),
        ('invoke.Context._set','invoke.config.DataProxy._set'),
        ('fabric.util.debug','util.debug'),
        ('invoke.terminals.pty_size','invoke.pty_size'),
        ('invoke.context.Context._run',"invoke.Context._run"),
        ('invoke.context.Context._sudo',"invoke.Context._sudo"),
        ('invoke.parser.argument.Argument',"invoke.Argument"),
        ('invoke.program.Program.core_args',"invoke.Program.core_args"),
        ('invoke.collection.Collection',"invoke.Collection"),
        ('invoke.program.Program.load_collection',"invoke.Program.load_collection"),
        ('invoke.program.Program.no_tasks_given',"invoke.Program.no_tasks_given"),
        ('invoke.program.Program.update_config',"invoke.Program.update_config"),
        ('invoke.collection.Collection',"invoke.Collection"),
        ('re.compile.findall','re.findall'),
        ('requests.sessions.Session.get','requests.Session.get'),
        ('argparse.ArgumentParser.add_argument','argparse.add_argument'),
        ('argparse.ArgumentParser.parse_args','argparse.parse_args'),
        ('threading.Thread.start','threading.start')
    ]
    f1_values = []
    threshold_list = []
    op = 'san'
    #op = 'merged'
    for threshold in range(0,101,10):
        print(f"threshold:{threshold}")
        tp, fn, fp, tn = 0,0,0,0
        # for name in project_name_list_1:
        #     input_file1 = "./data/{}_data.csv".format(name)
        #     input_file2 = "./data/{}_result.csv".format(name)
        #     solve(name,input_file1, input_file2, thread)
        f1 = 0 
        for name in project_name_list_1:
            input_file1 = "../src/data/{}_data.csv".format(name)
            input_file2 = "../src/data/{}_result.csv".format(name)
            if name == "autojump":
                replace_list.append(('bin.',''))
            elif name == "face_classification":
                replace_list.append(('src.',''))
            solve(name,input_file1, input_file2, threshold, replace_list, skip_list, pre_clean, all_or_EA=1,op=op) 
            precision_i, recall_i, f1_i = main(name, replace_list, skip_list, pre_clean, precision_write=0, recall_write=0, Ae_or_pycg=1, all_or_EA=1)
            f1 += f1_i
            if name == "autojump":
                replace_list.pop()
            elif name == "face_classification":
                replace_list.pop()
            
        #project_name_list_2 = []
        for name in project_name_list_2:
            input_file1 = "../src/data/{}_data.csv".format(name)
            input_file2 = "../src/data/{}_result.csv".format(name)
            solve(name,input_file1, input_file2, threshold, replace_list, skip_list, pre_clean, all_or_EA=0,op=op)
            precision_i, recall_i, f1_i = main(name, replace_list, skip_list, pre_clean, precision_write=0, recall_write=0, Ae_or_pycg=1, all_or_EA=0) 
            f1 += f1_i
        f1 = f1 / (len(project_name_list_1) + len(project_name_list_2))
        f1_values.append(f1)
        threshold_list.append(threshold)

    print("Thresholds:", threshold_list)
    #f1_values[16] = 84.6623
    print("F1 Values:", f1_values)
    # import matplotlib.pyplot as plt

    # # 推荐风格：论文级别美观（matplotlib 自带）
    # plt.style.use('seaborn-whitegrid')

    # plt.figure(figsize=(10, 6))

    # plt.plot(
    #     threshold_list,
    #     f1_values,
    #     marker='o',
    #     markersize=8,
    #     linewidth=2.0,
    #     label='F1 Score'
    # )

    # # 坐标轴刻度字体
    # plt.xticks(threshold_list, fontsize=12)
    # plt.yticks(fontsize=12)

    # # 坐标轴标签和标题
    # plt.xlabel('Threshold (%)', fontsize=14)
    # plt.ylabel('F1 Score (%)', fontsize=14)
    # #plt.title('F1 Score vs. Threshold', fontsize=16)

    # # 网格细化风格
    # plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)

    # # 图例
    # plt.legend(fontsize=12, loc='best')

    # # 优化边框（常见学术风格）
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # # 让布局更紧凑
    # plt.tight_layout()

    # plt.savefig('deepseek_threshold_f1_analysis.png', dpi=600, bbox_inches='tight')
    # plt.show()
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline

    plt.style.use('seaborn-whitegrid')

    # 原始数据
    x = np.array(threshold_list)
    y = np.array(f1_values)

    # ---- 样条插值：生成平滑曲线 ----
    x_smooth = np.linspace(x.min(), x.max(), 300)   # 300 个点用于平滑
    spline = make_interp_spline(x, y, k=3)          # k=3 = cubic spline
    y_smooth = spline(x_smooth)

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_smooth,
        y_smooth,
        linewidth=2.5,
        label='F1 Score'
    )

    # 坐标轴刻度
    plt.xticks(x, fontsize=12)  # 保留原阈值作为刻度
    plt.yticks(fontsize=12)

    plt.xlabel('Threshold (%)', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    #plt.title('F1 Score vs. Threshold (Smoothed Curve)', fontsize=16)

    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    plt.legend(fontsize=12, loc='best')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('deepseek_threshold_f1_analysis_smooth.png', dpi=600, bbox_inches='tight')
    plt.show()








            

