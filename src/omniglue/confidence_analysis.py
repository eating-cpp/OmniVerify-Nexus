import numpy as np

class ConfidenceAnalysis:
    """
    该类用于对所有文件的置信度分布进行分析，具体包括对每个文件的置信度分布进行积分、排序，
    并将积分结果归一化到 0 到 1 区间，同时对相应的文件名进行排序。

    参数:
    all_confidence_distributions (list): 包含所有文件置信度分布的列表，每个元素是一个表示置信度分布的数组。
    all_subfolder_names (list): 包含所有文件对应文件夹名称的列表。
    """
    def __init__(self, all_confidence_distributions, all_subfolder_names, all_match_confidence,all_match_kp0,all_match_kp1):
        # 初始化所有文件的置信度分布
        self.all_confidence_distributions = all_confidence_distributions
        # 初始化所有文件对应的文件夹名称
        self.all_subfolder_names = all_subfolder_names
        self.all_match_confidence = all_match_confidence
        self.all_match_kp0 = all_match_kp0
        self.all_match_kp1 = all_match_kp1

    def integrate_and_sort(self):
        """
        对每个文件的置信度分布进行积分，按照积分结果对置信度分布和文件名进行排序，
        并将积分结果归一化到 0 到 1 区间。

        返回:
        tuple: 包含两个元素，第一个元素是排序后的置信度分布列表，第二个元素是排序后的文件名列表。
        """
        # 对每个文件的置信度分布进行积分
        # 使用列表推导式和 np.trapz 函数计算每个置信度分布的积分
        integrals = [np.trapz(distribution) for distribution in self.all_confidence_distributions]
        print(f">\t置信度分布积分结果: {integrals}")

        # 获取排序后的索引
        # np.argsort 函数返回数组元素从小到大排序后的索引，[::-1] 表示反转索引顺序，即从大到小排序
        sorted_indices = np.argsort(integrals)
        sorted_integrals=np.sort(integrals)

        # 根据排序后的索引对置信度分布和文件名进行排序
        # 使用列表推导式和排序后的索引重新排列置信度分布列表
        sorted_distribution_list = [self.all_confidence_distributions[i] for i in sorted_indices]
        # 使用列表推导式和排序后的索引重新排列文件名列表
        sorted_subfolder_name_list = [self.all_subfolder_names[i] for i in sorted_indices]
        sotred_all_match_confidence = [self.all_match_confidence[i] for i in sorted_indices]
        sorted_all_match_kp0 = [self.all_match_kp0[i] for i in sorted_indices]
        sorted_all_match_kp1 = [self.all_match_kp1[i] for i in sorted_indices]

        # 打印排序后的置信度分布和文件名列表
        print(f">\t排序后的置信度积分: {sorted_integrals}")
        print(f">\t排序后的文件名列表: {sorted_subfolder_name_list}")
        
        mean, std_dev, iqr, lower_bound, upper_bound, outliers, outliers_index=self.calculate_statistics(sorted_integrals)
        print(f">\t平均值: {mean}")
        print(f">\t标准差: {std_dev}")
        print(f">\t均值加两倍标准差: {mean+2*std_dev}")
        print(f">\t四分位距: {iqr}")
        print(f">\t四分位距下界: {lower_bound}")
        print(f">\t四分位距上界: {upper_bound}")
        print(f">\t离群值: {outliers}")
        # print(f">\t离群值索引: {outliers_index}")
        # 根据离群值索引输出对应的文件名
        for i in range(len(outliers_index)):
            print(f">\t离群值索引: {outliers_index[i]}, 对应的文件名: {sorted_subfolder_name_list[outliers_index[i]]}")
        
        # # 归一化积分值到 0 到 1 区间
        # # 将积分结果转换为 numpy 数组
        # integrals = np.array(integrals)
        # # 检查积分结果的最大值和最小值之差是否不为 0
        # if integrals.max() - integrals.min() != 0:
        #     # 如果不为 0，则进行归一化操作，将积分结果缩放到 0 到 1 区间
        #     normalized_integrals = (integrals - integrals.min()) / (integrals.max() - integrals.min())
            
        # else:
        #     # 如果为 0，则直接使用原始积分结果，因为无法进行缩放
        #     normalized_integrals = integrals
        
        # output_normalized_integrals = [normalized_integrals[i] for i in sorted_indices]
        # # 打印归一化后的积分结果和对应的文件名
        # for i in range(len(sorted_integrals)):
        #     print(f">\t归一化后的置信度积分: {sorted_subfolder_name_list[i]}: {output_normalized_integrals[i]}")
            
            
        return mean, std_dev, iqr, lower_bound, upper_bound, outliers, outliers_index,sorted_distribution_list, sorted_subfolder_name_list,sorted_integrals,sotred_all_match_confidence,sorted_all_match_kp0,sorted_all_match_kp1

    def calculate_statistics(self,data):
    # 计算标准差
        mean = np.mean(data)
        std_dev = np.std(data)

        # 计算四分位距
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        # 计算四分位距的上下界
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 找出离群的数
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        # 同时也找到离群的文件名
        outliers_index = [i for i in range(len(data)) if data[i] < lower_bound or data[i] > upper_bound]

        return mean, std_dev, iqr, lower_bound, upper_bound, outliers, outliers_index

 
# # 示例数据
# data = [112, 96.5, 90.5, 76.5, 75, 59]
# std_dev, iqr, lower_bound, upper_bound, outliers = calculate_statistics(data)

