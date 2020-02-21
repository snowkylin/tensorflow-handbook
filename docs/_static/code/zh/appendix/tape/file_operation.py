with open('test.txt', 'w') as f:    # open() 是文件资源的上下文管理器，f 是文件资源对象
    f.write('hello world')
f.write('another string')   # 报错，因为离开上下文环境时，资源对象 f 被其上下文管理器所释放