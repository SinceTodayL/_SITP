## Some Points 



所有代码的 `random_state` (随机值种子) 要固定一致，以保证实验结果的可重复性

同时，需要验证不同的 `random_state` ，验证结果不会偏差太大，以保证**模型的稳定性**

本项目后期所有代码保证 `random_state` 均设置为 **42**，同时设置为 36、100 验证模型的稳定性



20250308

或许可以用传统异常检测的方式，如 https://github.com/buhuixiezuowendelihua/Anomaly-Detection 中提到的 $3\theta$ 准则，以及箱线图（四分位数）的方式，做一个比对，然后可以验证自己机器学习/深度学习方法的有效性



分类问题，本质上就是要最大化后验概率



还有多元高斯模型/单一高斯模型，也可以与深度学习方式作比较