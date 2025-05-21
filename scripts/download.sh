wget https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz 

# 创建目标目录（如果不存在）
mkdir -p ~/data/KuaiRand-1K

# 解压到指定目录
tar -xzvf KuaiRand-1K.tar.gz -C ~/data/KuaiRand-1K

# 删除原始压缩包
rm KuaiRand-1K.tar.gz