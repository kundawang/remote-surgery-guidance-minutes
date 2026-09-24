import os

# 测试不依赖外部数据库，导入 app 模块前先把数据库指向内存 SQLite
os.environ.setdefault("DATABASE_URL", "sqlite://")
