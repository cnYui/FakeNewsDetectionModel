import nltk
import os

# 设置NLTK数据目录
nltk_data_dir = '/root/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# 设置下载目录
download_dir = nltk_data_dir

print(f"NLTK数据将下载到: {download_dir}")

# 下载所需资源
try:
    nltk.download('wordnet', download_dir=download_dir)
    nltk.download('omw-1.4', download_dir=download_dir)
    print("NLTK资源下载成功")
except Exception as e:
    print(f"下载NLTK资源时出错: {e}")
    
# 验证资源是否可用
try:
    from nltk.corpus import wordnet
    synsets = wordnet.synsets('test')
    print(f"WordNet可用，'test'的同义词集合数量: {len(synsets)}")
except Exception as e:
    print(f"无法使用WordNet: {e}")
