# [小猿口算]自动化答题

该程序并未采用OCR的方案进行识别，而是采用了图形相似度匹配数字，因此仅支持`5以内数的比大小`。

至于为什么没采用OCR，是因为OCR识别会出错，这个代码最开始修改自[小猿口算脚本-CSDN博客](https://blog.csdn.net/qq_45910820/article/details/142795826)。原作者所使用的ddddocr经常将`1`识别为`7`，导致了多次错误，改用匹配图形后就没出现过这个问题。

当然，这个脚本相对抓包的方案来说并不快，因此看到这里的你可以把代码中的视觉部分改用抓包，这份代码的特色不过是全自动化罢了。



# 必备工具

- mumu模拟器
- 安装必要的库

```bash
pip install numpy Pillow opencv-python pyautogui
```



# TODO

有空的话我会把结束后的点击部分从pyautogui改用adb实现(这只是图一乐，所以也有可能不改)



# 鸣谢

参考代码https://blog.csdn.net/qq_45910820/article/details/142795826