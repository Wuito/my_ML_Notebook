filename = 'output2.csv';
% 使用 csvwrite 函数保存数组到 CSV 文件
% csvwrite(filename, dcmoter_wm1);
plot(dcmoter_wm)
x_data1 = dcmoter_wm(: , 1);
y_data1 = dcmoter_wm(: , 2);
