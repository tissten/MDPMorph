import numpy as np
import argparse

# 添加参数解析
parser = argparse.ArgumentParser(description="Generate random test suites.")
parser.add_argument("--num_suites", type=int, default=50, help="Number of test suites to generate.")
parser.add_argument("--num_cases", type=int, default=100, help="Number of test cases per suite.")
args = parser.parse_args()

num_sets = args.num_suites
num_cases = args.num_cases


fmt = ["%.6f"] * 13 + ["%.1f"] * 11


for i in range(0, num_sets + 1):
    test_cases = []  
    for _ in range(num_cases):
        case = []

        for k in range(0, 13):
            value = np.random.normal(0, 0.1)
            case.append(value)

        case.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        test_cases.append(case)
    

    test_cases = np.array(test_cases)
    

    file_name = f"T{i}.txt"
    

    np.savetxt(file_name, test_cases, fmt=fmt, delimiter=" ")
    
    print(f"Test suite {file_name} has been generated.")



