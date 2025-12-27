import csv
import re
import json

project_name_list_1 = ["asciinema","autojump","fabric","face_classification","Sublist3r"]
project_name_list_1 = ["asciinema","autojump"] #测试集
#project_name_list_1 = ["fabric","face_classification","Sublist3r"] #训练集
project_name_list_2 = ['bpytop','furl','rich_cli','sqlparse','sshtunnel','textrank4zh']
project_name_list_2 = ['sqlparse','sshtunnel','textrank4zh'] #测试集
#project_name_list_2 = ['bpytop','furl','rich_cli'] #训练集
project_list_id=[1,3,4,7,8,11,12,13,14,15,16,22,23,28,29,32,33,35,36,38,45,48,53,56,58]


def calculate_metrics(true_edges, predicted_edges):
    """
    计算 TP、FN、FP、TN，并返回它们，用于后续绘制 AUC。
    """
    # 真正例（TP）：真实和预测都存在的边
    tp = true_edges & predicted_edges

    # 假负例（FN）：真实中存在，但预测中不存在的边
    fn = true_edges - predicted_edges

    # 假正例（FP）：预测中存在，但真实中不存在的边
    fp = predicted_edges - true_edges

    # ---------- NEW ADD: 计算 TN ----------
    total_true_edges = len(true_edges)
    total_pred_edges = len(predicted_edges)

    # 估计全集大小，设全集为两集合边的并集
    union_edges = true_edges | predicted_edges
    tn = len(union_edges) - (len(tp) + len(fn) + len(fp))

    return len(tp), len(fn), len(fp), tn   # << 修改返回内容



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
        for r in replace_list:
            source = source.replace(r[0],r[1])
 
        if all_or_EA == 0:
            if not source.startswith('<') and not source.startswith(name):
                continue
        for target in targets:

            if "error" in target.lower():
                continue

            if target.split('.')[0] in skip_list:
                continue

            for r in replace_list:
                target = target.replace(r[0],r[1])

            if all_or_EA == 0:
                if not target.startswith('<') and not target.startswith(name):
                    continue   
            edges.add((source, target))

    return edges

def main(name, replace_list, skip_list, pre_clean=[], precision_write=0, recall_write=0, Ae_or_pycg=1, all_or_EA=1):

    true_json_path = "../ground-truth-cgs/{}.json".format(name)

    if Ae_or_pycg == 1:
        #print("{} Ae:".format(name))
        predicted_json_path = "../Ae_data/{}.json".format(name)
    elif Ae_or_pycg == 0:
        #print("{} PyCG:".format(name))
        predicted_json_path = "../PyCG_data/{}.json".format(name)
    elif Ae_or_pycg == 2:
        #print("{} Depends:".format(name))
        predicted_json_path = "../Depends_data/{}.json".format(name)

    true_edges = load_json_1(name, true_json_path, replace_list, skip_list, pre_clean, all_or_EA)
    predicted_edges = load_json_1(name, predicted_json_path, replace_list, skip_list, pre_clean, all_or_EA)

    # 拿到 TP, FN, FP, TN
    tp, fn, fp, tn = calculate_metrics(true_edges, predicted_edges)

    # 输出
    # print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

    return tp, fn, fp, tn   # << 返回四个指标


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
        return tp, fn, fp, tn


    else :
        true_json_path = "../ground-truth-cgs/{}.json".format(name)
        true_edges = load_json_1(name, true_json_path, replace_list, skip_list, pre_clean, all_or_EA)
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
        tp, fn, fp, tn = 0,0,0,0
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
                    if (caller_candidates[i]["caller"], caller_candidates[i]["callee"]) in true_edges:
                        result[i] = 1
                        tp += 1
                    else:
                        result[i] = 0
                        fp += 1
                else :
                    if (caller_candidates[i]["caller"], caller_candidates[i]["callee"]) in true_edges:
                        result[i] = 0
                        fn += 1
                    else:
                        result[i] = 0
                        tn += 1
            return tp, fn, fp, tn

            

    # for i, call in enumerate(caller_candidates):

    #     caller, candidate = call["caller"], call["callee"]
    #     yes_total = result[i]
    #     name = project_name_idx[i]
    #     if yes_total:
    #         if caller in project_data[name]:
    #             project_data[name][caller].append(candidate)  # 将 candidate 添加到原列表
    #             # 去重
    #             project_data[name][caller] = list(set(project_data[name][caller]))
    #         else:
    #             project_data[name][caller] = []
    #             project_data[name][caller].append(candidate)  # 如果 caller 不存在，直接插入


    # filename_with_extension = project_json_path[name].split('/')[-1]
    # with open("../Ae_data/{}".format(filename_with_extension), "w", encoding="utf-8") as file:
    #     json.dump(project_data[name], file, indent=4, ensure_ascii=False)


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
    tpr_values = []
    fpr_values = []
    threshold_list = []
    op = 'san'
    #op = 'merged'
    for threshold in range(0,111,10):
        print(f"threshold:{threshold}")
        tp, fn, fp, tn = 0,0,0,0
        # for name in project_name_list_1:
        #     input_file1 = "./data/{}_data.csv".format(name)
        #     input_file2 = "./data/{}_result.csv".format(name)
        #     solve(name,input_file1, input_file2, thread) 
        for name in project_name_list_1:
            input_file1 = "../src/data/{}_data.csv".format(name)
            input_file2 = "../src/data/{}_result.csv".format(name)
            if name == "autojump":
                replace_list.append(('bin.',''))
            elif name == "face_classification":
                replace_list.append(('src.',''))
            tp_i, fn_i, fp_i, tn_i = solve(name,input_file1, input_file2, threshold, replace_list, skip_list, pre_clean, all_or_EA=1,op=op) 
            tp += tp_i
            fn += fn_i
            fp += fp_i
            tn += tn_i
            if name == "autojump":
                replace_list.pop()
            elif name == "face_classification":
                replace_list.pop()
            
        #project_name_list_2 = []
        for name in project_name_list_2:
            input_file1 = "../src/data/{}_data.csv".format(name)
            input_file2 = "../src/data/{}_result.csv".format(name)
            tp_i, fn_i, fp_i, tn_i = solve(name,input_file1, input_file2, threshold, replace_list, skip_list, pre_clean, all_or_EA=0,op=op) 
            tp += tp_i
            fn += fn_i
            fp += fp_i
            tn += tn_i

        # 计算 TPR / FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)
        threshold_list.append(threshold)

    # import matplotlib.pyplot as plt
    # from sklearn.metrics import auc    
    # # 计算真正的 AUC
    # roc_auc = auc(fpr_values, tpr_values)

    # # 绘制 ROC
    # plt.figure()
    # plt.plot(fpr_values, tpr_values)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    # plt.show()
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    # 计算AUC
    roc_auc = auc(fpr_values, tpr_values)

    # 绘制ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, color='darkorange', lw=2, 
            label='DeepSeek ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('DeepSeek: ROC Analysis Across Confidence Thresholds')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 保存文件
    plt.savefig('deepseek_threshold_roc_analysis.png', dpi=1000, bbox_inches='tight')
    plt.show()
            

