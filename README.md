## 植株提取拼接工具



### 使用方法

#### 1.命令行

```shell
pip install -r requirements.txt
python main.py
```

#### 2.Windows 可执行文件

```shell
#打包
pyinstaller --name PlantConcat -D main.py
cp BG202388-42D.xml dist/PlantConcat/_internal/
cp -r run dist/PlantConcat/_internal/
cp -r font dist/PlantConcat/_internal/

```

可执行文件存储在/dist/PlantConcat下，点击运行即可

如果需要修改裁剪位置，请打开[Make Sense](https://www.makesense.ai/)根据新的文件创建裁剪配置文件，并在启动后修改`配置文件->切割位置配置文件：`
