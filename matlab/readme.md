# MATLAB - ML

## 数据拟合

给出的 `dc-moter-output.csv`是用simulink模拟出来的小型直流有刷电机12V输入下的转速输出曲线。

模拟的电机传递函数是一个带反馈的线性系统，理论上用双指数函数拟合会百分百拟合。

可以使用matlab对元数据做双指数函数参数拟合

$$
f(x) = a\times e ^{b\cdot x } + c\times e ^{d\cdot x }
$$

可以使用matlab的cftool工具箱进行双指数函数拟合
