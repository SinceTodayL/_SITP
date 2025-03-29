import torch

'''
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())'
'''


import time


print("当前 PyTorch 线程数:", torch.get_num_threads())

A = torch.randn(5000, 5000)
B = torch.randn(5000, 5000)

start_time = time.time()

C = A * B  # 向量化计算，一步完成所有加法

end_time = time.time()
print("向量化计算耗时:", end_time - start_time, "秒")   

torch.set_num_threads(1)  
start_time = time.time()
C = A @ B  # 矩阵乘法
end_time = time.time()
print("单线程计算时间:", end_time - start_time, "秒")

torch.set_num_threads(4) 
start_time = time.time()
C = A @ B
end_time = time.time()
print("多线程计算时间:", end_time - start_time, "秒")

A = A.cuda()
B = B.cuda()

start_time = time.time()

C = A * B  # 向量化计算，一步完成所有加法

end_time = time.time()
print("向量化计算耗时:", end_time - start_time, "秒")   

torch.set_num_threads(1) 
start_time = time.time()
C = A @ B  
end_time = time.time()
print("单线程计算时间:", end_time - start_time, "秒")

torch.set_num_threads(4)  
start_time = time.time()
C = A @ B
end_time = time.time()
print("多线程计算时间:", end_time - start_time, "秒")