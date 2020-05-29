with open('test.txt', 'w') as f:    # open() 是文件資源的上下文管理器，f 是文件資源使用者
    f.write('hello world')
f.write('another string')   # 報錯，因為離開上下文環境時，資源使用者 f 被其上下文管理器所釋放